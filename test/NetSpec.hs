module NetSpec (main, spec) where

import Test.Hspec
import Test.QuickCheck

import Network

-- `main` is here so that this module can be run from GHCi on its own.  It is
-- not needed for automatic spec discovery.
main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "sigmoid" $ do
    it "is 0 at -inf" $ do
      sigmoid (-1/0 :: Double) `shouldBe` 0
    it "is 1 at +inf" $ do
      sigmoid (1/0 :: Double) `shouldBe` 1
    it "is 0.5 at 0" $ do
      sigmoid (0 :: Double) `shouldBe` 0.5
    it "is between 0 and 1 for all real numbers" $ 
      property $ \x -> (sigmoid x) <= (1.0 :: Double) && (sigmoid x) >= (0.0 :: Double)

  describe "sigmoid'" $ do
    it "is 0 at -inf" $ do
      sigmoid' (-1/0 :: Double) `shouldBe` 0
    it "is 0 at +inf" $ do
      sigmoid' (1/0 :: Double) `shouldBe` 0
    it "is 0.25 at 0" $ do
      sigmoid' (0 :: Double) `shouldBe` 0.25
    it "is always positive" $
      property $ \x -> (sigmoid' x) >= (0.0 :: Double)
    


