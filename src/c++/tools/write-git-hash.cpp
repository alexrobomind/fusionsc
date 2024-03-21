#include <kj/io.h>
#include <kj/main.h>
#include <kj/filesystem.h>
#include <kj/debug.h>
#include <iostream>

#include <cstdlib>

struct MainCls {
	kj::ProcessContext& context;
	
	kj::String hashFile = kj::heapString("hashTmpFile");
	kj::String outputFile = nullptr;
	kj::String varName = nullptr;
	kj::String includeFile = nullptr;
	
	MainCls(kj::ProcessContext& context):
		context(context)
	{}
	
	// Parameter parsers
	
	bool setOutput(kj::StringPtr output) {
		outputFile = kj::heapString(output);
		return true;
	}

	bool setVarName(kj::StringPtr val) {
		varName = kj::heapString(val);
		return true;
	}

	bool setIncludeFile(kj::StringPtr val) {
		includeFile = kj::heapString(val);
		return true;
	}
	
	bool setTmp(kj::StringPtr val) {
		hashFile = kj::heapString(val);
		return true;
	}
	
	// Main code
	
	bool main() {
		auto fs = kj::newDiskFilesystem();
		auto inPath = fs -> getCurrentPath().evalNative(hashFile);
		
		// Run git rev-parse and redirect output into the hash file
		auto retCode = std::system(kj::str("git rev-parse HEAD > ", hashFile).cStr());
		
		kj::String gitHash = kj::heapString("<unknown>");
		
		// Important: We abort with OK if command failed
		if(retCode == 0) {
			{
				auto inFile = fs -> getRoot().openFile(inPath);
				gitHash = inFile -> readAllText();
				
				// Remove newlines & other stuff
				kj::Vector<char> charBuf;
				for(auto c : gitHash) {
					if(c == '\n' ||c == '\r')
						continue;
					charBuf.add(c);
				}
				
				gitHash = kj::heapString(charBuf.releaseAsArray());
			}
			
			auto retCode = std::system(kj::str("git status -s -uno > ", hashFile).cStr());
			KJ_REQUIRE(retCode == 0, "git rev-parse HEAD succeeded, but git status failed");
			
			{
				auto inFile = fs -> getRoot().openFile(inPath);
				auto statusData = inFile -> readAllText();
				
				size_t modifiedFiles = 0;
				for(char c : statusData) {
					if(c == '\n') ++modifiedFiles;
				}
				KJ_LOG(INFO, modifiedFiles);
				
				if(modifiedFiles != 0) {
					gitHash = kj::str(mv(gitHash), "+");
				}
			}
		} else {
			KJ_LOG(WARNING, "git rev-parse HEAD failed, commit hash unknown");
		}
		
		
		auto outPath = fs -> getCurrentPath().evalNative(outputFile);
		
		
		KJ_LOG(INFO, gitHash);
			
		kj::String output = kj::str(
			"#include <kj/string.h>\n",
			"#include <", includeFile, ">\n",
			"\n"
			"kj::StringPtr ", varName, " = \"", gitHash, "\"_kj;"
		);
		
		KJ_IF_MAYBE(pOutFile, fs -> getRoot().tryOpenFile(outPath, kj::WriteMode::MODIFY)) {
			kj::String existing = (**pOutFile).readAllText();
			
			if(existing != output) {
				KJ_LOG(INFO, "Hash file changed");
				(**pOutFile).writeAll(output);
			} else {
				KJ_LOG(INFO, "Hash file unchanged");
			}
		} else {
			auto outFile = fs -> getRoot().openFile(outPath, kj::WriteMode::CREATE);
			outFile -> writeAll(output);
			KJ_LOG(INFO, "Hash file created");
		}
		
		return true;
	}
	
	kj::MainFunc getMain() {		
		return kj::MainBuilder(context, "Embeds files into C++ files", "Extractor for git hashes")
			.expectArg("output", KJ_BIND_METHOD(*this, setOutput))
			.expectArg("varName", KJ_BIND_METHOD(*this, setVarName))
			.expectArg("include", KJ_BIND_METHOD(*this, setIncludeFile))
			.addOptionWithArg({'t', "tempFile"}, KJ_BIND_METHOD(*this, setTmp), "<tempfile>", "Temporary file used to buffer git outputs")
			.callAfterParsing(KJ_BIND_METHOD(*this, main))
			.build()
		;
	}
};

KJ_MAIN(MainCls);
