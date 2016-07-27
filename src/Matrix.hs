module Matrix where

import System.Random

data Size = Size { width :: Int, height :: Int } deriving Show

data Padding = NoPadding | ValuePadding Float deriving Show

class Matrix a where
    (>*<), (=*=), (=+=), (=-=), (=/=) :: a -> a -> a
    (<*=), (<+=), (<-=), (</=) :: Float -> a -> a
    convolve :: Padding -> a -> a -> a
    randnMat :: (RandomGen g) => g -> Size -> a
    fromList :: Size -> [Float] -> a
    toList :: a -> [Float]
    size :: a -> Size
    
    
