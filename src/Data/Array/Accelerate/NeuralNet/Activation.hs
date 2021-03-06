module Data.Array.Accelerate.NeuralNet.Activation where

import Data.Array.Accelerate as A

class Activation act where
  act_apply :: (Shape ix) => act -> Acc (Array ix Float) -> Acc (Array ix Float)
  act_delta :: (Shape ix) => act -> Acc (Array ix Float) -> Acc (Array ix Float)


data Sigmoid = Sigmoid deriving Show

-- | O(n)
instance Activation Sigmoid where
  act_apply _ = A.map (\z -> 1 / (1 + exp (-z)))
  act_delta _ = A.map (\z -> let s = 1 / (1 + exp (-z))
                         in s * (1 - s)) 

data ReLU = ReLU deriving Show

instance Activation ReLU where
  act_apply _ = A.map (\z -> (z <* 0) ? (1, z))
  act_delta _ = A.map (\z -> (z <* 0) ? (0, 1))
