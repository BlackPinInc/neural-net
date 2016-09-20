module Data.Array.Accelerate.NeuralNet.Loader where

import Prelude as P
import qualified Data.ByteString.Lazy as BS
import Data.Binary.Get
import Text.Printf
import Control.Applicative ((<$>))
import System.FilePath.Posix ((</>))
import Data.Array.Accelerate as A


loadMnistArrays :: FilePath -> IO ([(Array DIM2 Int, Int)], [(Array DIM2 Int, Int)])
loadMnistArrays folder = do
  (test, train) <- loadMnistData folder
  let newTest = (\(x,y) -> (fromList (Z:.28:.28) x, y)) <$> test
  let newTrain = (\(x,y) -> (fromList (Z:.28:.28) x, y)) <$> train
  return (newTest, newTrain)

loadMnistData :: FilePath -> IO ([([Int], Int)], [([Int], Int)])
loadMnistData folder = do
  testLabels <- loadIdxLabels $ folder </> "t10k-labels.idx1-ubyte"
  testImages <- loadIdxImages $ folder </> "t10k-images.idx3-ubyte"
  trainingLabels <- loadIdxLabels $ folder </> "train-labels.idx1-ubyte"
  trainingImages <- loadIdxImages $ folder </> "train-images.idx3-ubyte"
  return (P.zip testImages testLabels, P.zip trainingImages trainingLabels)

loadIdxLabels :: FilePath -> IO [Int]
loadIdxLabels fname = do
  labelsString <- BS.readFile fname
  let labels = runGet readLabels labelsString
  return $ P.fromIntegral <$> labels

readLabels = do
  magic <- getWord32be
  if magic == 0x0801 then return () else error "Wrong Magic Number"
  numItems <- getWord32be
  labels <- mapM (const getWord8) [1..numItems]
  return labels

loadIdxImages :: FilePath -> IO [[Int]]
loadIdxImages fname = P.map (P.map P.fromIntegral) <$> runGet readImages <$> BS.readFile fname

readImages = do
  magic <- getWord32be
  if magic == 0x0803 then return () else error "Wrong Magic Number"
  numItems <- getWord32be
  numRows <- getWord32be
  numCols <- getWord32be
  images <- mapM (const (mapM (const getWord8) [1..numRows * numCols])) [1..numItems]
  return images
