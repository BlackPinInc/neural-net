module Data.Array.Accelerate.NeuralNet.Layer.StaticLayers (
  StaticLayer,
  ActivationLayer, mkActivationLayer, mkSigmoidLayer,
  NormalizeLayer, mkNormalizeLayer,
  ReshapeLayer, mkReshapeLayer
) where

import Prelude as P
import Data.Array.Accelerate 
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.Arithmetic.LinearAlgebra as T

import Data.Array.Accelerate.NeuralNet.Layer.Types
import Data.Array.Accelerate.NeuralNet.Activation


type StaticLayer = Layer ()


-- ActivationLayer -- 

-- | Activation layer applies an activation function to the input array.
type ActivationLayer s = StaticLayer s s Float Float s s

-- | Create an activation layer with the given activation function.
mkActivationLayer :: (Activation a, Shape s, Slice s) 
                  => a                  -- ^ Activation function.
                  -> Exp s              -- ^ Shape of the input and output
                  -> ActivationLayer s  -- ^ The output Layer.
mkActivationLayer a size = Layer { param = ()
                                 , inputSize = size
                                 , outputSize = size
                                 , feedForward = \_ i -> a `act_apply` i
                                 , feedBack = (\_ i p -> ( ()
                                                         , A.zipWith (*) p (a `act_delta` i)
                                                         ))
                                 , removeError = const $ const ()
                                 , combineError = const $ const ()
                                 , layerCost = const 0
                                 }

-- | Create a Sigmoid Activation Layer.
mkSigmoidLayer :: (Shape s, Slice s) => Exp s -> ActivationLayer s
mkSigmoidLayer = mkActivationLayer Sigmoid


type NormalizeLayer s i o = StaticLayer s s i o s s 

mkNormalizeLayer :: (Elt i, Integral i, IsIntegral i, Shape s, Slice s)
                 => i
                 -> Exp s
                 -> NormalizeLayer s i Float
mkNormalizeLayer i size = layer
  where i' = P.fromIntegral i
        layer = Layer { param = ()
                      , inputSize = size
                      , outputSize = size
                      , feedForward = const $ A.map ((/i').A.fromIntegral)
                      , feedBack = const $ (\x p -> ((), A.map (/i') p)) 
                      -- TODO: Heres hoping I never use this 
                      -- I really dont trust this math
                      -- Don't underestimate how little I trust this math
                      , removeError = const $ const ()
                      , combineError = const $ const ()
                      , layerCost = const 0
                      }

type ReshapeLayer s t i o = StaticLayer s s t t i o

mkReshapeLayer :: (Shape sh1, Shape sh2, Elt t) 
               => Exp sh1
               -> Exp sh2
               -> ReshapeLayer s t sh1 sh2 
mkReshapeLayer inSize outSize = Layer { param = ()
                                      , inputSize = inSize
                                      , outputSize = outSize
                                      , feedForward = const $ reshape outSize
                                      , feedBack = const $ (\x p -> ((), p))
                                      , removeError = const $ const ()
                                      , combineError = const $ const ()
                                      , layerCost = const 0
                                      }
 
type FlattenLayer s t i = ReshapeLayer s t i DIM1

mkFlattenLayer inSize = mkReshapeLayer inSize (lift $ Z:.(shapeSize inSize))
