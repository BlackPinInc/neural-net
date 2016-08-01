module NetSpec (main, spec) where

import Test.Hspec
import Test.QuickCheck
import Network
import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter as I

-- `main` is here so that this module can be run from GHCi on its own.  It is
-- not needed for automatic spec discovery.
main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "Sigmoid" $ do
    let neginf = (-1/0) -- -inf
    let posinf = (1/0) -- +inf
    let zero = 0 -- 0
    let testSingleton f x = indexArray (run (f (unit x))) Z

    it "is 0 at -inf" $ do
      testSingleton (apply Sigmoid) neginf `shouldBe` 0
    it "is 1 at inf" $ do
      testSingleton (apply Sigmoid) posinf `shouldBe` 1
    it "is 0.5 at 0" $ do
      testSingleton (apply Sigmoid) zero `shouldBe` 0.5

    it "has a delta of 0 at -inf" $ do
      testSingleton (delta Sigmoid) neginf `shouldBe` 0
    it "has a delta of 0 at inf" $ do
      testSingleton (delta Sigmoid) posinf `shouldBe` 0
    it "has a delta of 0.25 at 0" $ do
      testSingleton (delta Sigmoid) zero `shouldBe` 0.25

