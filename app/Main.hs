module Main where

import Prelude as P
import Data.Array.Accelerate.NeuralNet 
import Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.CUDA as I



main :: IO ()
main = do
  (test, train) <- loadMnistArrays "/usr/local/share/mnist"
  layer <- mkNormalMatMulLayer 0 1.0 (28*28) 10
  let each input = feedForward layer (param layer) (A.map ((/255.0).A.fromIntegral) $ flatten input)
  print $ head $ I.stream each $ P.map P.fst train
