{-# LANGUAGE TypeOperators #-}
module Data.Array.Accelerate.NeuralNet.Util where

import Prelude (($), IO, return, (-))
import qualified Prelude as P
import Control.Applicative ((<$>))
import System.Random.MWC.Distributions
import Data.Array.Accelerate as A
import Data.Array.Accelerate.System.Random.MWC

normalArray :: (Shape sh) => Float -> Float -> sh -> IO (Acc (Array sh Float))
normalArray m s sh = do
  let generator _ g = P.realToFrac <$> normal (P.realToFrac m) (P.realToFrac s) g
  gen <- create
  init <- randomArrayWith gen generator sh
  return $ use init

