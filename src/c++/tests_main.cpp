#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>

int main( int argc, char* argv[] ) {
  int result = Catch::Session().run( argc, argv );

  return result;
}

TEST_CASE( "Test subsystem is operational", "" ) {
	SUCCEED();
}