module Main where

import Data.Array.Accelerate.NeuralNet 
import Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.Interpreter as I


main :: IO ()
main = do
  let layer = mkUnitMatMulLayer 1 1000 5000
  let input = enumFromN (index1 1000) 1
  let outputAcc = feedForward layer (param layer) input
  let output = I.run $ A.sum $ outputAcc
  print output
  
