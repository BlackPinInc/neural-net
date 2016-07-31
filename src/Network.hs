{-# LANGUAGE FlexibleContexts, TupleSections #-}
module Network where

import Data.Maybe (isJust, fromJust)
import Data.Random.Normal
import System.Random
import Data.Array.Accelerate as A

type Tensor ix = Array ix Float

sigmoid :: (Shape ix) => Acc (Tensor ix) -> Acc (Tensor ix)
sigmoid = A.map (\z -> 1 / (1 + exp (-z)))

sigmoid' :: (Shape ix) => Acc (Tensor ix) -> Acc (Tensor ix)
sigmoid' = A.map (\z -> let s = 1 / (1 + exp (-z)) 
                        in s * (1 - s))

class Layer layer where
  feedForward :: (Shape inputShape, Shape outputShape) 
              => layer -- ^ Current Layer in the network 
              -> Acc (Tensor inputShape) -- ^ Input tensor
              -> Acc (Tensor outputShape) -- ^ Output tensor

  feedBack    :: (Shape sh1, Shape sh2) 
              => layer -- ^ Current Layer in the network
              -> Acc (Tensor sh1) -- ^ The derivative of the previous weight matrix
              -> (Acc (Tensor sh1), -- ^ The derivative of this weight matrix
                  Acc (Tensor sh1), -- ^ The error in this weight matrix 
                  Acc (Tensor sh2)) -- ^ The error in this bias matrix

  removeError :: (Shape sh1, Shape sh2)
              => layer -- ^ Current input Layer 
              -> Acc (Tensor sh1) -- ^ Error in the biases of the layer
              -> Acc (Tensor sh2) -- ^ Error in the weights of the layer
              -> Float -- ^ Eta for the error
              -> Float -- ^ Lambda for the error
              -> Int -- ^ The total number of things `n` lol im not sure
              -> Int -- ^ The size of the minibatch
              -> layer -- ^ The output layer (because we are modifying the layer)

  cost        :: layer
              -> Float 
  


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

