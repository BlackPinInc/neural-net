{-# LANGUAGE FlexibleContexts, TupleSections, TypeFamilies, ExistentialQuantification #-}
module Network where

import Data.Maybe (isJust, fromJust)
import Data.Random.Normal
import System.Random
import Data.Array.Accelerate as A
import Data.Array.Accelerate.Arithmetic.LinearAlgebra


type Tensor ix = Array ix Float

data Sigmoid = Sigmoid deriving Show

instance Activation Sigmoid where
  apply _ = A.map (\z -> 1 / (1 + exp (-z)))
  delta a x = A.map (\z -> 1 * (1 - z)) s
    where s = apply a x


sigmoid :: (Shape ix) => Acc (Tensor ix) -> Acc (Tensor ix)
sigmoid = A.map (\z -> 1 / (1 + exp (-z)))

sigmoid' :: (Shape ix) => Acc (Tensor ix) -> Acc (Tensor ix)
sigmoid' = A.map (\z -> let s = 1 / (1 + exp (-z)) 
                        in s * (1 - s))
class Activation act where
  apply :: (Shape ix) => act -> Acc (Tensor ix) -> Acc (Tensor ix)
  delta :: (Shape ix) => act -> Acc (Tensor ix) -> Acc (Tensor ix)

class Layer layer where
  type InputShape layer :: *
  type OutputShape layer :: *
  type WeightShape layer :: *
  type BiasShape layer :: *
  feedForward :: layer -- ^ Current Layer in the network 
              -> Acc (Tensor (InputShape layer)) -- ^ Input tensor
              -> Acc (Tensor (OutputShape layer)) -- ^ Output tensor

  feedBack    :: layer -- ^ Current Layer in the network
              -> Acc (Tensor (WeightShape layer)) -- ^ The derivative of the previous weight matrix
              -> ( Acc (Tensor (WeightShape layer)) -- ^ The derivative of this weight matrix
                 , Acc (Tensor (WeightShape layer)) -- ^ The error in this weight matrix 
                 , Acc (Tensor (BiasShape layer)) -- ^ The error in this bias matrix
                 )

  removeError :: layer -- ^ Current input Layer 
              -> Acc (Tensor (BiasShape layer)) -- ^ Error in the biases of the layer
              -> Acc (Tensor (WeightShape layer)) -- ^ Error in the weights of the layer
              -> Float -- ^ Eta for the error
              -> Float -- ^ Lambda for the error
              -> Int -- ^ The total number of things `n` lol im not sure
              -> Int -- ^ The size of the minibatch
              -> layer -- ^ The output layer (because we are modifying the layer)

  cost        :: layer
              -> Float 
  
data FullyConnectedLayer = forall a . Activation a => FullyConnectedLayer 
                              { fcl_weights :: Acc (Tensor DIM2)
                              , fcl_biases :: Acc (Tensor DIM1)
                              , fcl_activation :: a
                              }

instance Layer FullyConnectedLayer where
  type InputShape FullyConnectedLayer = DIM1
  type OutputShape FullyConnectedLayer = DIM1
  type WeightShape FullyConnectedLayer = DIM2
  type BiasShape FullyConnectedLayer = DIM1
  feedForward (FullyConnectedLayer w b a) input = a `apply` z
    where wx = multiplyMatrixVector w input
          z = A.zipWith (+) wx b
{-
  feedBack (FullyConnectedLayer w b a) prev = 
    where sp = a `delta` 
-}

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

