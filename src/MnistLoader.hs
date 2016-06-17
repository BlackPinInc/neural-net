module MnistLoader where

import qualified Data.ByteString.Lazy as BS
import Data.Binary.Get
import Text.Printf
import Data.Matrix
import TrainingSet

loadData = do
  testLabels <- loadLabels "t10k-labels.idx1-ubyte"
  testImages <- loadImages "t10k-images.idx3-ubyte"
  trainingLabels <- loadLabels "train-labels.idx1-ubyte"
  trainingImages <- loadImages "train-images.idx3-ubyte"
  return (zip testImages testLabels, zip trainingImages trainingLabels)

loadDataWrapper :: IO ([TrainingSet], [TrainingSet])
loadDataWrapper = do
  (testData, trainingData) <- loadData
  let convertImage img = fromList (length img) 1 $ map ((/255).fromIntegral) img
  let convertLabel lab = fromList 10 1 $ replicate (fromIntegral lab) 0 ++ [1] ++ replicate (9 - fromIntegral lab) 0
  let testData' = map (\(img, lab) -> (TrainingSet (convertImage img) (convertLabel lab))) testData
  let trainingData' = map (\(img, lab) -> (TrainingSet (convertImage img) (convertLabel lab))) trainingData
  return (testData', trainingData')

loadLabels fname = do
  labelsString <- BS.readFile fname
  let labels = runGet readLabels labelsString
  return labels


readLabels = do
  magic <- getWord32be
  if magic == 0x0801 then return () else error "Wrong Magic Number"
  numItems <- getWord32be
  labels <- mapM (const getWord8) [1..numItems]
  return labels

loadImages fname = runGet readImages <$> BS.readFile fname

readImages = do
  magic <- getWord32be
  if magic == 0x0803 then return () else error "Wrong Magic Number"
  numItems <- getWord32be
  numRows <- getWord32be
  numCols <- getWord32be
  images <- mapM (const (mapM (const getWord8) [1..numRows * numCols])) [1..numItems]
  return images
