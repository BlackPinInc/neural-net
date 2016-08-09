{-# LANGUAGE FlexibleContexts, TupleSections, TypeFamilies, ExistentialQuantification, TypeOperators #-}
module Data.Array.Accelerate.NeuralNet.Layer where

import Prelude
import qualified Prelude as P
import Control.Applicative ((<$>))
import Data.Array.Accelerate as A hiding ((++))
import qualified System.Random.MWC.Distributions as R
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.Array.Sugar as S
import qualified Data.Array.Accelerate.System.Random.MWC as R
import qualified Data.Array.Accelerate.Arithmetic.LinearAlgebra as T

import Data.Array.Accelerate.NeuralNet.Activation

type Tensor ix = Acc (Array ix Float) 

sizeError i o = error $ "Input size: " ++ show i ++ " does not match output size: " ++ show o

data Layer w i o = Layer { inputSize :: Exp i
                         , outputSize :: Exp o
                         , param :: w
                         , feedForward :: w -> Tensor i -> Tensor o
                         , feedBack    :: w -> Tensor i -> Tensor DIM1 -> (w, Tensor DIM1)
                         , removeError :: w -> w -> w
                         }


-- MatMulLayer --
                               
type MatMulLayer = Layer (T.Matrix Z Float) DIM1 DIM1

mkMatMulLayer :: T.Matrix Z Float -> Int -> Int -> MatMulLayer
mkMatMulLayer init inSize outSize = layer
  where weights = init -- TODO: randoms instread of 1
        ff w input = w `T.multiplyMatrixVector` input
        fb w input prevDeriv = (dw, T.vectorFromColumn d)
          where dw = d `T.multiplyMatrixMatrix` (T.transpose $ T.columnFromVector input)
                d = (T.transpose w) `T.multiplyMatrixMatrix` (T.columnFromVector prevDeriv)
        re = A.zipWith (-) 
        layer = Layer { inputSize = index1 $ lift inSize
                      , outputSize = index1 $ lift outSize
                      , param = weights
                      , feedForward = ff
                      , feedBack = fb
                      , removeError = re
                      }

mkUnitMatMulLayer :: Float -> Int -> Int -> MatMulLayer
mkUnitMatMulLayer i inSize outSize = mkMatMulLayer init inSize outSize
  where init = fill (index2 (lift outSize) (lift inSize)) $ lift i

mkNormalMatMulLayer :: Float -> Float -> Int -> Int -> IO MatMulLayer
mkNormalMatMulLayer m s inSize outSize = do
  let generator _ g = realToFrac <$> R.normal (realToFrac m) (realToFrac s) g
  gen <- R.create
  init <- R.randomArrayWith gen generator (Z:.outSize:.inSize)
  return $ mkMatMulLayer (use init) inSize outSize


-- BiasLayer -- 

type BiasLayer = Layer (T.Vector Z Float) DIM1 DIM1

mkBiasLayer :: T.Vector Z Float -> Int -> BiasLayer
mkBiasLayer init size = layer
  where biases = init -- TODO: randoms?
        ff b input = A.zipWith (+) input b
        fb _ _ p = (p, p)
        re = A.zipWith (-) 
        layer = Layer { inputSize = index1 $ lift size
                      , outputSize = index1 $ lift size
                      , param = biases
                      , feedForward = ff
                      , feedBack = fb
                      , removeError = re
                      }

mkUnitBiasLayer :: Float -> Int -> BiasLayer
mkUnitBiasLayer i size = mkBiasLayer (fill (index1 $ lift size) $ lift i) size 

mkNormalBiasLayer :: Float -> Float -> Int -> IO BiasLayer
mkNormalBiasLayer m s size = do
  let generator _ g = realToFrac <$> R.normal (realToFrac m) (realToFrac s) g
  gen <- R.create
  init <- R.randomArrayWith gen generator (Z:.size)
  return $ mkBiasLayer (use init) size


-- ActivationLayer -- 

type ActivationLayer = Layer () DIM1 DIM1

mkActivationLayer :: (Activation a) => a -> Int -> ActivationLayer 
mkActivationLayer a size = Layer { param = ()
                                 , inputSize = index1 $ lift size
                                 , outputSize = index1 $ lift size
                                 , feedForward = \_ i -> a `apply` i
                                 , feedBack = (\_ i p -> ( ()
                                                         , A.zipWith (*) p (a `delta` i)
                                                         ))
                                 , removeError = const $ const ()
                                 }

mkSigmoidLayer :: Int -> ActivationLayer
mkSigmoidLayer = mkActivationLayer Sigmoid


-- Network -- 

type Network w1 w2 i o = Layer (w1, w2) i o

mkNetwork :: (Shape o1, Shape i2)
          => Layer w1 i o1
          -> Layer w2 i2 o
          -> Network w1 w2 i o
mkNetwork layer1 layer2 = net
  where p = (param layer1, param layer2)
        ff (w1, w2) input = feedForward layer2 w2 reshaped
          where out = feedForward layer1 w1 input
                reshaped = reshape (lift $ inputSize layer2) out
        fb (w1, w2) input prevDeriv = ((nw1, nw2), out1)
          where input2 = reshape (inputSize layer2) $ feedForward layer1 w1 input
                (nw2, out2) = feedBack layer2 w2 input2 prevDeriv
                (nw1, out1) = feedBack layer1 w1 input out2
        re (w1, w2) (nw1, nw2) = ( removeError layer1 w1 nw1
                                 , removeError layer2 w2 nw2
                                 )
        net = Layer { param = p
                    , inputSize = (inputSize layer1)
                    , outputSize = (outputSize layer2)
                    , feedForward = ff
                    , feedBack = fb
                    , removeError = re
                    }
                
type FullyConnectedLayer = Layer (T.Matrix Z Float, T.Vector Z Float) DIM1 DIM1

mkFullyConnectedLayer :: (Activation a) => a -> Int -> Int -> IO FullyConnectedLayer 
mkFullyConnectedLayer a i o = do
  matMulLayer <- mkNormalMatMulLayer 0 (sqrt (P.fromIntegral i)) i o
  biasLayer <- mkNormalBiasLayer 0 1 o
  let actLayer = mkActivationLayer a o
  let p = (param matMulLayer, param biasLayer)
  let ff (w, b) input = let wx = feedForward matMulLayer w input
                            z = feedForward biasLayer b wx
                            a = feedForward actLayer () z
                        in a
  let fb (w, b) input prevDeriv = let wx = feedForward matMulLayer w input
                                      z = feedForward biasLayer b wx
                                      (_, d1) = feedBack actLayer () z prevDeriv
                                      (nb, d2) = feedBack biasLayer b wx d1
                                      (nw, newDeriv) = feedBack matMulLayer w input d2
                                  in ((nw, nb), newDeriv)
  let re (w, b) (nw, nb) = ( removeError matMulLayer w nw
                           , removeError biasLayer b nb
                           )
  let net = Layer { param = p
                  , inputSize = index1 $ lift i
                  , outputSize = index1 $ lift o
                  , feedForward = ff
                  , feedBack = fb
                  , removeError = re
                  }
  return net

type MultilayerPerceptron = Layer [(T.Matrix Z Float, T.Vector Z Float)] DIM1 DIM1


 
 


