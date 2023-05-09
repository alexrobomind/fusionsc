@0x975b6b5e47dc52dd;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc::lvn");

using Java = import "java.capnp";
$Java.package("org.fsc.lvn");
$Java.outerClassname("LocalVatNetwork");

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

struct JoinKeyPart {
	secretPiece @0 : Data;
	secretHash @1 : Data;
	numPieces @2 : UInt64;
}

struct JoinResult {
	vatKey @0 : UInt64;
}

const initialVatId : VatId = (key = 0);