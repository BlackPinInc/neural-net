{-# LANGUAGE ScopedTypeVariables, TypeFamilies, AllowAmbiguousTypes #-}
module Data.Array.Accelerate.NeuralNet.CostFunction where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Arithmetic.LinearAlgebra as T

(<>) = (A.++)

argmax :: forall ix a. (Shape ix, Elt ix, Ord a, Elt a, IsScalar a) => Acc (Array ix a) -> Exp ix
argmax arr = A.snd exp 
  where ixs = generate (shape arr) id
        zipped = A.zip arr ixs
        myMax a b = (ax >* bx) ? (a, b)
          where (ax, ai) = unlift a :: (Exp a, Exp ix)
                (bx, bi) = unlift b :: (Exp a, Exp ix)
        scalar = fold1All myMax zipped 
        exp = the scalar 
        

class CostFunction a where
  cost_apply :: a -> T.Vector Z Float -> Exp Int -> Exp Float 
  cost_apply c v n = cost_apply' c v z
    where z = fill (index1 (n-1)) 0 <> fill (index1 1) 1 <> fill (index1 (size v - n)) 0
  cost_apply' :: a -> T.Vector Z Float -> T.Vector Z Float -> Exp Float
  cost_apply' c v z = cost_apply c v n
    where (Z:.n) = unlift $ argmax z
  cost_delta :: a -> T.Vector Z Float -> T.Vector Z Float -> T.Vector Z Float

data CrossEntropy = CrossEntropy deriving Show

instance CostFunction CrossEntropy where
  cost_apply' c a y = the $ A.sum $ A.zipWith (\a y -> -y*log a - (1-y)*log(1-a)) a y
  cost_delta c = A.zipWith (\a y -> -y / a + (1-y)/(1-a)) 

data QuadraticCost = QuadraticCost deriving Show

instance CostFunction QuadraticCost where
  cost_apply' c a y = the $ A.sum $ A.zipWith (\a y -> (a - y) * (a - y)) a y
  cost_delta c = A.zipWith (\a y -> a - y)  

