#include <catch2/catch_test_macros.hpp>

#include <cupnp-test.capnp.h>
#include <cupnp-test.capnp.cu.h>

#include <capnp/message.h>

using cupnp::test::TestStruct;

TEST_CASE("Exporting to GPU") {
	capnp::MallocMessageBuilder msg;
	
	auto root = msg.initRoot<TestStruct>();
	
	root.setU32(15);
	root.setD64(2.0);
	root.getUnion1().setVal2(5);
	
	root.initLu32(4);
	for(size_t i = 0; i < root.getLu32().size(); ++i) root.getLu32().set(i, i);
	
	root.initLb(7);
	for(size_t i = 0; i < root.getLb().size(); ++i) root.getLb().set(i, i % 3 == 0);
	
	cupnp::HostMessage hostMsg(msg.getSegmentsForOutput());
	cupnp::Message cupnpMsg = hostMsg.message;
	
	cupnp::CupnpVal<TestStruct> cupnpRoot = cupnp::messageRoot<TestStruct>(cupnpMsg);
	auto union1 = cupnpRoot.getUnion1();
	
	REQUIRE(root.getU32() == cupnpRoot.getU32());
	REQUIRE(root.getD64() == cupnpRoot.getD64());
	
	REQUIRE(!union1.hasVal1());
	REQUIRE(union1.hasVal2());
	
	REQUIRE(union1.getVal1() == 3);
	REQUIRE(union1.getVal2() == root.getUnion1().getVal2());
	
	REQUIRE(root.getLu32().size() == cupnpRoot.getLu32().size());
	for(size_t i = 0; i < root.getLu32().size(); ++i)
		REQUIRE(root.getLu32()[i] == cupnpRoot.getLu32()[i]);
	
	REQUIRE(root.getLu32Def().size() == cupnpRoot.getLu32Def().size());
	KJ_LOG(WARNING, cupnpRoot.getLu32Def().size());
	for(size_t i = 0; i < root.getLu32Def().size(); ++i) {
		REQUIRE(root.getLu32Def()[i] == cupnpRoot.getLu32Def()[i]);
		KJ_LOG(WARNING, cupnpRoot.getLu32Def()[i]);
	}
	
	REQUIRE(root.getLb().size() == cupnpRoot.getLb().size());
	for(size_t i = 0; i < root.getLb().size(); ++i) {
		REQUIRE(root.getLb()[i] == cupnpRoot.getLb()[i]);
		KJ_LOG(WARNING, i, cupnpRoot.getLb()[i]);
	}
	
	REQUIRE_THROWS(cupnpRoot.mutateLu32Def());
}