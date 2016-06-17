module TrainingSet where

import Data.Matrix

data TrainingSet = TrainingSet { input :: Matrix Double
                               , output :: Matrix Double
                               } deriving (Show, Eq)


