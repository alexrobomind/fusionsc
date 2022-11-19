@0x975b6b5e47dc52dd;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc::lvn");

struct VatId {
	key @0 : UInt64;
}

struct ProvisionId {
	providerKey @0 : UInt64;
	id @1 : UInt64;
}

struct RecipientId {
	recipientKey @0 : UInt64;
	id @1 : UInt64;
}

struct ThirdPartyCapId {
	thirdPartyKey @0 : UInt64;
	id @1 : UInt64;
}

struct JoinResult{]