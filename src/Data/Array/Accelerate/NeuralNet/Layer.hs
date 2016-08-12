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
import Data.Array.Accelerate.NeuralNet.CostFunction

data Layer weightType inDeriv outDeriv iType oType iShape oShape = 
     Layer { inputSize :: Exp iShape
           , outputSize :: Exp oShape
           , param :: weightType
           , feedForward :: weightType 
                         -> Acc (Array iShape iType)
                         -> Acc (Array oShape oType)
           , feedBack    :: weightType 
                         -> Acc (Array iShape iType) 
                         -> Acc (Array inDeriv Float)
                         -> (weightType, Acc (Array outDeriv Float))
           , removeError :: weightType -> weightType -> weightType
           , combineError :: weightType -> weightType -> weightType
           }


-- MatMulLayer --
                               
type MatMulLayer = Layer (T.Matrix Z Float) DIM1 DIM1 Float Float DIM1 DIM1

mkMatMulLayer :: T.Matrix Z Float -> Int -> Int -> MatMulLayer 
mkMatMulLayer init inSize outSize = layer
  where weights = init -- TODO: randoms instread of 1
        ff w input = T.multiplyMatrixVector w input
        fb w input prevDeriv = (dw, T.vectorFromColumn d)
          where dw = T.multiplyMatrixMatrix d (T.transpose $ T.columnFromVector input)
                d = (T.transpose w) `T.multiplyMatrixMatrix` (T.columnFromVector prevDeriv)
        re = A.zipWith (-) 
        ce = A.zipWith (+)
        layer :: MatMulLayer
        layer = Layer { inputSize = index1 $ lift inSize
                      , outputSize = index1 $ lift outSize
                      , param = weights
                      , feedForward = ff
                      , feedBack = fb
                      , removeError = re
                      , combineError = ce
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

type BiasLayer = Layer (T.Vector Z Float) DIM1 DIM1 Float Float DIM1 DIM1

mkBiasLayer :: T.Vector Z Float -> Int -> BiasLayer
mkBiasLayer init size = layer
  where biases = init -- TODO: randoms?
        ff b input = A.zipWith (+) input b
        fb _ _ p = (p, p)
        re = A.zipWith (-) 
        ce = A.zipWith (+) 
        layer = Layer { inputSize = index1 $ lift size
                      , outputSize = index1 $ lift size
                      , param = biases
                      , feedForward = ff
                      , feedBack = fb
                      , removeError = re
                      , combineError = ce
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

type ActivationLayer s = Layer () s s Float Float s s

mkActivationLayer :: (Activation a, Shape s, Slice s) 
                  => a -> Exp s -> ActivationLayer s
mkActivationLayer a size = Layer { param = ()
                                 , inputSize = size
                                 , outputSize = size
                                 , feedForward = \_ i -> a `act_apply` i
                                 , feedBack = (\_ i p -> ( ()
                                                         , A.zipWith (*) p (a `act_delta` i)
                                                         ))
                                 , removeError = const $ const ()
                                 , combineError = const $ const ()
                                 }

mkSigmoidLayer :: (Shape s, Slice s) => Exp s -> ActivationLayer s
mkSigmoidLayer = mkActivationLayer Sigmoid


-- CostLayer -- 



-- Network -- 

type TupleLayer w1 w2 iDeriv oDeriv 
                iType oType iSize oSize = Layer (w1, w2) iDeriv oDeriv 
                                                iType oType iSize oSize

mkNetwork :: Layer w1 d oDeriv1 iType1 t iSize1 s
          -> Layer w2 iDeriv2 d t oType2 s oSize2
          -> TupleLayer w1 w2 iDeriv2 oDeriv1 iType1 oType2 iSize1 oSize2
mkNetwork layer1 layer2 = net
  where p = (param layer1, param layer2)
        ff (w1, w2) input = feedForward layer2 w2 out
          where out = feedForward layer1 w1 input
        fb (w1, w2) input prevDeriv = ((nw1, nw2), out1)
          where input2 = feedForward layer1 w1 input
                (nw2, out2) = feedBack layer2 w2 input2 prevDeriv
                (nw1, out1) = feedBack layer1 w1 input out2
        re (w1, w2) (nw1, nw2) = ( removeError layer1 w1 nw1
                                 , removeError layer2 w2 nw2
                                 )
        ce (w1, w2) (nw1, nw2) = ( combineError layer1 w1 nw1
                                 , combineError layer2 w2 nw2
                                 )
        net = Layer { param = p
                    , inputSize = (inputSize layer1)
                    , outputSize = (outputSize layer2)
                    , feedForward = ff
                    , feedBack = fb
                    , removeError = re
                    , combineError = ce
                    }
                
type FullyConnectedLayer = Layer (T.Matrix Z Float, T.Vector Z Float) DIM1 DIM1 Float Float DIM1 DIM1

mkFullyConnectedLayer :: (Activation a) => a -> Int -> Int -> IO FullyConnectedLayer 
mkFullyConnectedLayer a i o = do
  matMulLayer <- mkNormalMatMulLayer 0 (sqrt (P.fromIntegral i)) i o
  biasLayer <- mkNormalBiasLayer 0 1 o
  let actLayer = mkActivationLayer a (index1 $ constant o)
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
  let ce (w, b) (nw, nb) = ( combineError matMulLayer w nw
                           , combineError biasLayer b nb
                           )
  let net = Layer { param = p
                  , inputSize = index1 $ lift i
                  , outputSize = index1 $ lift o
                  , feedForward = ff
                  , feedBack = fb
                  , removeError = re
                  , combineError = ce
                  }
  return net

type MultiLayerNetwork w d t s = Layer [w] d d t t s s

nullNetwork :: MultiLayerNetwork w d t s
nullNetwork = Layer { param = []
                    , feedForward = const id
                    , feedBack = \_ _ p -> ([],p)
                    , removeError = const $ const $ []
                    , combineError = const $ const $ []
                    }

consNetwork :: Layer w d d t t s s -> MultiLayerNetwork w d t s -> MultiLayerNetwork w d t s
consNetwork layer net = net
  where ff (w:ws) input = feedForward net ws $ feedForward layer w input
        -- This looks crazy like theres no base case for the recursion but there isnt
        -- because this isnt recursive.
        -- it calls `feedForward` in `net` not in the current network!
        -- This code is blowing my fucking mind.
        fb (w:ws) input prevDeriv = (nw:nws, d2)
          where act = feedForward layer w input 
                (nws, d1) = feedBack net ws act prevDeriv
                (nw, d2) = feedBack layer w input d1
        -- This code runs `feedForward` all the way down the chain then comes back up with 
        -- the backpropagation. This is because the recursive step is between the individual steps.
        re (w:ws) (nw:nws) = (removeError layer w nw : removeError net ws nws)
        ce (w:ws) (nw:nws) = (combineError layer w nw : combineError net ws nws)

        net = Layer { param = (param layer) : (param net)
                    , inputSize = (inputSize layer) 
                    , outputSize = if (P.null $ param net) then (outputSize layer) else (outputSize net)
                    , feedForward = ff
                    , feedBack = fb
                    , removeError = re
                    , combineError = ce
                    }
                
mkMultiLayerNetwork :: [Layer w d d t t s s] -> MultiLayerNetwork w d t s 
mkMultiLayerNetwork [] = nullNetwork
mkMultiLayerNetwork (x:xs) = x `consNetwork` mkMultiLayerNetwork xs

                
data Network w s = forall c . CostFunction c => 
                 Network { innerLayer :: Layer w DIM1 DIM1 Float Float s DIM1
                         , costFunction :: c
                         }

netFeedForward :: Network w s -> [Acc (Array s Float)] -> [Acc (Array DIM1 Float)]
netFeedForward net = P.map (feedForward layer (param layer)) 
  where layer = innerLayer net


netBackprop :: Network w s -> (Acc (Array s Float), Int) -> w
netBackprop (Network layer cf) (arr, n) = nab
  where output = feedForward layer (param $ layer) arr
        prevDeriv = cf `cost_delta'` output $ (constant n)
        (nab, deriv) = feedBack layer (param $ layer) arr prevDeriv
 
netUpdateMiniBatch :: Network w s -> [(Acc (Array s Float), Int)] -> Network w s
netUpdateMiniBatch net batch = newNet
  where weights = P.map (netBackprop net) batch
        newWeight = P.foldl1 (combineError (innerLayer net)) weights
        newNet = net { innerLayer = (innerLayer net) { param = newWeight } }

