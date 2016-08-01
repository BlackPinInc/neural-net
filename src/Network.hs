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
  delta _ = A.map (\z -> let s = 1 / (1 + exp (-z))
                         in s * (1 - s)) 


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
  data InitParams layer :: *

  initialize  :: InitParams layer
              -> InputShape layer
              -> OutputShape layer
              -> layer

  feedForward :: layer
              -> Acc (Tensor (InputShape layer))
              -> Acc (Tensor (OutputShape layer))

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
  
data FullyConnectedLayer = FullyConnectedLayer 
                              { fcl_weights :: Acc (Tensor DIM2)
                              , fcl_biases :: Acc (Tensor DIM1)
                              }

instance Layer FullyConnectedLayer where
  type InputShape FullyConnectedLayer = DIM1
  type OutputShape FullyConnectedLayer = DIM1
  type WeightShape FullyConnectedLayer = DIM2
  type BiasShape FullyConnectedLayer = DIM1
  data InitParams FullyConnectedLayer = FCLDefault

  initialize opt inSize outSize = FullyConnectedLayer w b 
    where (Z:.x) = inSize
          (Z:.y) = outSize
          w = fill (lift (Z:.x:.y)) 1 -- TODO: double-check the arrangement of this Mat
                               -- TODO: Lol also make random instead of just 1
                               -- I didnt have internet when I wrote this.
          b = fill (lift (Z:.y)) 1 -- TODO: We might just want to leave this here
                            -- I have seen people just using 1 for their biases
                            -- So maybe this is advantagous 
                            -- Or maybe we pass that as an argument
                            -- TODO: Look up a program that finds TODOs in files
                            -- Or maybe make it?
  feedForward (FullyConnectedLayer w b) input = z
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

