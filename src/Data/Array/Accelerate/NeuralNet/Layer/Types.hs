module Data.Array.Accelerate.NeuralNet.Layer.Types (
  Layer(..)
) where

import Prelude as P
import Control.Applicative ((<$>))
import Data.Array.Accelerate as A

-- | Layer data type
-- Can be recombined recursively to produce larger and larger layers.
-- 
-- * `weightType` The type of the contained weight parameter.
-- For simple layers like MatMul and bias this parameter
-- is an Acc (Array x y) but for more complicated ones it 
-- it can be a tuple or a list of Acc (Array x y) 
-- * `inDeriv` The Shape of the incoming derivative. The derivative
-- should come from the next layer not the one before.
-- Thats been confusing me the whole time. Pretty much
-- all of the layers have this set to DIM1. 
-- * `outDeriv` The Shape of the outgoing derivative.  The derivative
-- produced by this layer and going into the previous layer.
-- Often this is the same as the incoming derivative Shape.
-- Pretty much all of the layers have this set to DIM1.
-- * `iType` The element type contained in the input Array.
-- This is usually Float but can be different.
-- * `oType` The element type contained in the output Array.
-- This is also usually Float but can be anything. 
-- * `iShape` The Shape of the input Array.
-- * `oShape` The Shape of the output Array.
data Layer weightType inDeriv outDeriv iType oType iShape oShape     
   = Layer { inputSize :: Exp iShape 
                      -- ^ The size of the input Array. This can be used by
                      -- the larger layers to check that they line up.
           , outputSize :: Exp oShape
                      -- ^ The size of the output Array.
           , param :: weightType
                      -- ^ The weight variable. If this is unused it can be set
                      -- to ().
           , feedForward :: weightType 
                         -> Acc (Array iShape iType)
                         -> Acc (Array oShape oType)
                      -- ^ The function to feed data through the layer. It feeds
                      -- forward and applies this layer to the input to get an
                      -- output.
           , feedBack    :: weightType 
                         -> Acc (Array iShape iType) 
                         -> Acc (Array inDeriv Float)
                         -> (weightType, Acc (Array outDeriv Float))
                      -- ^ The function to feed the derivative back through the
                      -- layer. It takes the input to the layer and the
                      -- derivative of the next layer and produces the
                      -- derivative of this layer with respect to the final output
                      -- and the derivative to be used by the previous layer.
           , removeError :: weightType -> weightType -> weightType
                      -- ^ Removes the weight derivate from the current weight
                      -- to produce the improved weight.
           , combineError :: weightType -> weightType -> weightType
                      -- ^ Combines two weight derivates to get the sum.
                      -- Basically allows its own instance of Monoid.
           , layerCost :: weightType -> Exp Float
           }


