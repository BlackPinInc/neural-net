module Data.Array.Accelerate.NeuralNet.Activation where

import Data.Array.Accelerate as A

data Sigmoid = Sigmoid deriving Show

instance Activation Sigmoid where
  apply _ = A.map (\z -> 1 / (1 + exp (-z)))
  delta _ = A.map (\z -> let s = 1 / (1 + exp (-z))
                         in s * (1 - s)) 

class Activation act where
  apply :: (Shape ix) => act -> Acc (Array ix Float) -> Acc (Array ix Float)
  delta :: (Shape ix) => act -> Acc (Array ix Float) -> Acc (Array ix Float)


