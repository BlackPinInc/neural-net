{-# LANGUAGE FlexibleContexts, TupleSections, TypeFamilies, ExistentialQuantification, TypeOperators #-}
module Network where

import Prelude hiding (zipWith)
import qualified Prelude as P
import Data.Maybe (isJust, fromJust)
import Data.Random.Normal
import System.Random
import Data.Array.Accelerate as A hiding (zipWith, (++))
import qualified Data.Array.Accelerate as A
import Data.Array.Accelerate.Arithmetic.LinearAlgebra as T
import qualified Data.Array.Accelerate.Array.Sugar as S


sizeError i o = error $ "Input size: " ++ show i ++ " does not match output size: " ++ show o

mulMatVec m v | indexHead (shape m) == indexHead (shape v) = multiplyMatrixVector m v
              | otherwise = error $ "Error with mat mul " ++ show (shape m) ++ " vs " ++ show (shape v)
type Tensor ix = Acc (Array ix Float)

data Sigmoid = Sigmoid deriving Show

instance Activation Sigmoid where
  apply _ = A.map (\z -> 1 / (1 + exp (-z)))
  delta _ = A.map (\z -> let s = 1 / (1 + exp (-z))
                         in s * (1 - s)) 

class Activation act where
  apply :: (Shape ix) => act -> Tensor ix -> Tensor ix
  delta :: (Shape ix) => act -> Tensor ix -> Tensor ix


data Layer w i o = Layer { inputSize :: Exp i
                         , outputSize :: Exp o
                         , param :: w
                         , feedForward :: w -> Tensor i -> Tensor o
                         , feedBack    :: w -> Tensor i -> Tensor DIM1 -> (w, Tensor DIM1)
                         , removeError :: w -> w -> w
                         }

type MatMulLayer = Layer (T.Matrix Z Float) DIM1 DIM1
                               
mkMatMulLayer :: Exp DIM1 -> Exp DIM1 
              -> MatMulLayer
mkMatMulLayer inSize outSize = layer
  where (Z:.x) = unlift inSize :: (Z:. Exp Int)
        (Z:.y) = unlift outSize :: (Z:. Exp Int)
        weights = fill (lift (Z:.y:.x)) 1 -- TODO: randoms instread of 1
        ff w input = w `multiplyMatrixVector` input
        fb w input prevDeriv = (dw, T.vectorFromColumn d)
          where dw = d `multiplyMatrixMatrix` (T.transpose $ T.columnFromVector input)
                d = (T.transpose w) `multiplyMatrixMatrix` (T.columnFromVector prevDeriv)
        re = A.zipWith (-) 
        layer = Layer { inputSize = inSize
                      , outputSize = outSize
                      , param = weights
                      , feedForward = ff
                      , feedBack = fb
                      , removeError = re
                      }

type BiasLayer = Layer (T.Vector Z Float) DIM1 DIM1

mkBiasLayer :: Exp DIM1
            -> BiasLayer
mkBiasLayer size = layer
  where biases = fill (lift size) 1 -- TODO: randoms?
        ff b input = A.zipWith (+) input b
        fb _ _ p = (p, p)
        re = A.zipWith (-) 
        layer = Layer { inputSize = size
                      , outputSize = size
                      , param = biases
                      , feedForward = ff
                      , feedBack = fb
                      , removeError = re
                      }

type ActivationLayer = Layer () DIM1 DIM1
mkActivationLayer :: (Activation a)
                  => a
                  -> Exp DIM1
                  -> ActivationLayer 
mkActivationLayer a size = Layer { param = ()
                                 , inputSize = size
                                 , outputSize = size
                                 , feedForward = \_ i -> a `apply` i
                                 , feedBack = (\_ i p -> ( ()
                                                         , A.zipWith (*) p (a `delta` i)
                                                         ))
                                 , removeError = const $ const ()
                                 }

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
                

-- Thing about what haskell does best.
-- is it algorithms or equations?
-- Avoid concrete types, use Num, Floating, or Functor instead of Matrix.
-- If you need to make your own class and have Matrix implement that class maybe you should.
-- I looked at the equations again last night and they will probably be easier to implement
-- in Haskell than the algorithms.
--
-- The feed forward algorithm has an equation that we can implement directly in Haskell.
-- No need to follow their code.  Or my own crappy code.  Blind leading the blind.
--
-- Take a look at the backprop equations(BP1-BP4) I can help you translate these directly
-- into haskell functions.  All of these equations have a direct haskell equivalent.
--
-- And you should look at this link for how to set up the unit testing.
-- https://github.com/kazu-yamamoto/unit-test-example
-- They use HSpec instead of HUnit but I would recomend HSpec because its more haskelly.
--
-- Lol sorry for all the text I am just very excited about this project and I want you to
-- have a better time on this than I did.

