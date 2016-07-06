module NetSpec (main, spec) where

import Test.Hspec
import Test.QuickCheck
import Data.Matrix
import Data.Vector (singleton)
import Network

-- `main` is here so that this module can be run from GHCi on its own.  It is
-- not needed for automatic spec discovery.
main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "sigmoid" $ do
    it "is 0 at -inf" $ do
      sigmoid (fromList 1 1 [-1/0] :: Matrix Double) `shouldBe` 0
    it "is 1 at +inf" $ do
      sigmoid (fromList 1 1 [1/0] :: Matrix Double) `shouldBe` 1
    it "is 0.5 at 0" $ do
      sigmoid (fromList 1 1 [0] :: Matrix Double) `shouldBe` 0.5
    -- it "is between 0 and 1 for all real numbers" $
    --   property $ \x -> (sigmoid x) <= (1.0 :: Matrix Double) && (sigmoid x) >= (0.0 :: Matrix Double)
    -- it "should work for matrices as well" $ do
    --   sigmoid (fromList 2 2 [1,1,1,1 :: Double]) == (fromList 2 2 (map sigmoid [1,1,1,1 :: Double]))
  --
  --
  describe "sigmoid'" $ do
    it "is 0 at -inf" $ do
      sigmoid' (fromList 1 1 [-1/0] :: Matrix Double) `shouldBe` 0
    it "is 0 at +inf" $ do
      sigmoid' (fromList 1 1 [1/0] :: Matrix Double) `shouldBe` 0
    it "is 0.25 at 0" $ do
      sigmoid' (fromList 1 1 [0] :: Matrix Double) `shouldBe` 0.25
  --   it "is always positive" $
  --     property $ \x -> (sigmoid' x) >= (0.0 :: Double)

  describe "mkNetwork" $ do
    it "makes a network" $ do
      mkNetwork [1,2,3,6,1] `shouldSatisfy` (const True)

    -- it "is 0 at +inf" $ do
    --   sigmoid' (fromList 1 1 [1/0] :: Matrix Double) `shouldBe` 0
    -- it "is 0.25 at 0" $ do
    --   sigmoid' (fromList 1 1 [0] :: Matrix Double) `shouldBe` 0.25
  --   it "is always positive" $
  --     property $ \x -> (sigmoid' x) >= (0.0 :: Double)
