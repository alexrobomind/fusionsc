#include "data-viewer.h"
#include "structio.h"

#include <kj/compat/url.h>

using capnp::Capability;
using capnp::AnyPointer;

namespace fsc {

namespace {

struct RefProxy : public DataRef<>::Server {
	const int index;

	RefProxy(int val) : index(val) {}
	
	virtual Maybe<int> getFd() { return index; }
};

int indexOf(Capability::Client clt) {
	KJ_IF_MAYBE(pIdx, capnp::ClientHook::from(mv(clt)) -> getFd()) {
		return *pIdx;
	}
	KJ_FAIL_REQUIRE("Reference is not a wrapped proxy");
}

LocalDataRef<AnyPointer> createWrapper(LocalDataRef<AnyPointer> local) {
	auto capTable = local.getCapTable();
	
	auto newCapTable = kj::heapArrayBuilder<Capability::Client>(capTable.size());
	for(auto i : kj::indices(capTable)) {
		newCapTable.add(kj::heap<RefProxy>(i));
	}
	
	return getActiveThread().dataService().publish<AnyPointer>(
		local.getMetadata(), local.forkRaw(), newCapTable.finish()
	);
}

kj::String escape(kj::StringPtr s) {
	kj::Vector<char> out;
	
	for(char c : s) {
		if(c == '<') {
			out.add('&'); out.add('l'); out.add('t'); out.add(';');
		} else if(c == '>') {
			out.add('&'); out.add('g'); out.add('t'); out.add(';');
		} else {
			out.add(c);
		}
	}
	
	out.add(0);
	
	return kj::String(out.releaseAsArray().releaseAsChars());
}

struct EscapingVisitor : public structio::SaveOptions::CapabilityStrategy, public structio::Visitor {
	structio::Visitor& backend;
	
	EscapingVisitor(structio::Visitor& newBackend) : backend(newBackend) {}
	
	void acceptNull() override { backend.acceptNull(); }
	void acceptInt(int64_t v) override { backend.acceptInt(v); }
	void acceptUInt(uint64_t v) override { backend.acceptUInt(v); }
	void acceptDouble(double d) override { backend.acceptDouble(d); }
	void acceptData(kj::ArrayPtr<const byte> d) override { backend.acceptData(d); }
	void acceptBool(bool b) override { backend.acceptBool(b); }
	void beginObject(Maybe<size_t> s) override { backend.beginObject(s); }
	void endObject() override { backend.endObject(); }
	void beginArray(Maybe<size_t> s) override { backend.beginArray(s); }
	void endArray() override { backend.endArray(); }

	bool done() override { return backend.done(); }
	
	void acceptString(kj::StringPtr s) override {
		backend.acceptString(escape(s));
	}
	
	void saveCapability(capnp::DynamicCapability::Client clt, structio::Visitor&, const structio::SaveOptions&, Maybe<kj::WaitScope&> maybeWs) const override {
		auto schema = clt.getSchema();
		auto id = schema.getProto().getId();

		kj::Vector<kj::StringTree> brandArgs;
		for(auto brandArg : schema.getBrandArgumentsAtScope(id)) {
			if(brandArg.isStruct()) {
				brandArgs.add(kj::strTree(brandArg.asStruct().getUnqualifiedName()));
			} else {
				brandArgs.add(kj::strTree("?"));
			}
		}

		auto brandName = kj::strTree(schema.getUnqualifiedName());
		if(brandArgs.size() > 0)
			brandName = kj::strTree(mv(brandName), "[", kj::StringTree(brandArgs.releaseAsArray(), ", "), "]");
		
		// Pass raw link to backend
		size_t idx = indexOf(clt);
		backend.acceptString(kj::str("<a href='", idx, "/show'>", mv(brandName), "</a>"));
	}
};

struct DataViewerImpl : public kj::HttpService {
	using SO = Warehouse::StoredObject;
	using Object = OneOf<DataRef<>::Client, Warehouse::Folder::Client, Warehouse::File<>::Client>;

	Object root;
	capnp::SchemaLoader& loader;

	DataViewerImpl(Object root, capnp::SchemaLoader& l) : root(root), loader(l) {}

	kj::Promise<void> request(
		kj::HttpMethod method, kj::StringPtr url, const kj::HttpHeaders& headers,
		kj::AsyncInputStream& requestBody, kj::HttpService::Response& response
	) override {		
		return kj::startFiber(
			1024 * 1024,
			[parsedUrl = kj::Url::parse(url, kj::Url::HTTP_REQUEST), &response, this](kj::WaitScope& ws) mutable {
				process(mv(parsedUrl), response, ws);
			}
		);
	}

	void process(kj::Url url, kj::HttpService::Response& response, kj::WaitScope& ws) {
		static kj::HttpHeaderTable DEFAULT_HEADERS;
		kj::HttpHeaders headers(DEFAULT_HEADERS);

		kj::Path path(url.path.releaseAsArray());
		if(path.size() == 0) {
			path = kj::Path("show");
		}

		kj::StringPtr op = path.basename()[0];
		kj::PathPtr objPath = path.parent();

		// Show operation
		Object o = get(objPath, ws);
		
		if(op == "show") {
			auto header = kj::strTree("<!DOCTYPE html><html><head><title>Contents</title></head><body>");
			auto headingBuilder = kj::heapArrayBuilder<kj::StringTree>(objPath.size() + 1);

			auto goUp = [&](size_t to) {
				kj::StringTree result;
				for(auto i : kj::range(0, objPath.size() - to)) {
					result = kj::strTree(mv(result), "../");
				}
				return kj::strTree(mv(result), "show").flatten();
			};

			for(auto i : kj::range(0, objPath.size() + 1)) {
				kj::StringPtr name = i == 0 ? kj::StringPtr("/") : objPath[i - 1];
				if(i < objPath.size()) {
					headingBuilder.add(kj::strTree("<a href='", goUp(i), "'>", name, "</a>"));
				} else {
					headingBuilder.add(kj::strTree(name));
				}
			}
			header = kj::strTree("<h1>Current location: ", mv(header), kj::StringTree(headingBuilder.finish(), " / "), "</h1>");

			if(path.size() > 1) {
				header = kj::strTree(mv(header), "<a href='../show'>..</a><br /><br />");
			}

			auto footer = "</body></html>"_kj;

			kj::VectorOutputStream os;
			if(o.is<DataRef<>::Client>()) {
				auto asRef = getActiveThread().dataService().download(o.get<DataRef<>::Client>()).wait(ws);
				writeRefBody(asRef, structio::Dialect::YAML, os);
			} else if(o.is<Warehouse::File<>::Client>()) {
				writeFileBody(os);
			} else {
				auto asFolder = o.get<Warehouse::Folder::Client>();
				writeFolderBody(asFolder, os, ws);
			}

			headers.set(kj::HttpHeaderId::CONTENT_TYPE, "text/html; charset=utf-8");
			auto aos = response.send(200, "OK", headers);
			aos -> write({ header.flatten().asBytes(), os.getArray(), footer.asBytes() }).wait(ws);
		} else if(op == "download") {
			// Download operation

			// Create in-memory file to save
			KJ_REQUIRE(o.is<DataRef<>::Client>(), "Can only download DataRef objects");

			auto file = kj::newInMemoryFile((kj::Clock&) getActiveThread().timer());
			getActiveThread().dataService().writeArchive(o.get<DataRef<>::Client>(), *file).wait(ws);

			headers.set(kj::HttpHeaderId::CONTENT_TYPE, "application/octet-stream");
			headers.add("Content-Disposition", kj::str("attachment; filename=\"", objPath[objPath.size() - 1], ".fsc\""));
			auto aos = response.send(200, "OK", headers);

			// Stream file into memory
			// Weird, this doesn't work because it apparently changes the size of the
			// backing store (and the file doesn't instantly close)
			// auto mapped = file -> mmap(0, file -> stat().size);
			auto mapped = file -> readAllBytes();
			aos -> write(mapped.begin(), mapped.size()).wait(ws);
		} else {
			KJ_FAIL_REQUIRE("Invalid operation", op);
		}
	}
	
	Object get(kj::PathPtr p, kj::WaitScope& ws) {
		Object result = root;

		auto handleSo = [&](Warehouse::StoredObject::Reader r) {
			if(r.isFolder()) {
				result = r.getFolder();
			} else if(r.isFile()) {
				result = r.getFile();
			} else {
				result = (DataRef<>::Client) r.getAsGeneric();
			}
		};
		
		while(p.size() > 0) {
			if(result.is<DataRef<>::Client>()) {
				auto asRef = result.get<DataRef<>::Client>();

				auto asInt = p[0].parseAs<uint32_t>();
				
				result = asRef
					.metaAndCapTableRequest()
					.send().wait(ws)
					.getTable()[asInt].castAs<DataRef<>>();

				p = p.slice(1, p.size());
			} else if(result.is<Warehouse::File<>::Client>()) {
				auto asFile = result.get<Warehouse::File<>::Client>();
				KJ_REQUIRE(p[0] == "contents", "Files only have a child 'contents'");

				handleSo(asFile.getAnyRequest().send().wait(ws));
				p = p.slice(1, p.size());
			} else {
				auto asFolder = result.get<Warehouse::Folder::Client>();
				
				auto req = asFolder.getRequest();
				req.setPath(p.toString());

				handleSo(req.send().wait(ws));
				break;
			}
		}

		return result;
	}

	//! Writes a body for a database folder
	void writeFolderBody(Warehouse::Folder::Client folder, kj::BufferedOutputStream& os, kj::WaitScope& ws) {
		auto lsResponse = folder.getAllRequest().send().wait(ws);

		auto response = kj::strTree(
			"Folder contents:<br /><br /><table><tr>"
			"<td>Name</td>"
			"<td>Content</td>"
			"</tr>"
		);
		for(auto entry : lsResponse.getEntries()) {
			auto val = entry.getValue();
			
			kj::String content;
			if(val.isUnresolved())
				content = kj::str("Unresolved");
			else if(val.isNullValue())
				content = kj::str("Null");
			else if(val.isException())
				content = kj::str("Error: ", kj::str(val.getException()));
			else if(val.isFolder())
				content = kj::str("Folder");
			else if(val.isFile())
				content = kj::str("Mutable File");
			else if(val.isDead())
				content = kj::str("Dead object");
			else if(val.isDataRef()) {
				if(val.getDataRef().getDownloadStatus().isFinished())
					content = kj::str("Data");
				else
					content = kj::str("Data [download in progress]");
			}
			
			kj::StringTree name = kj::strTree(entry.getName());
			if(val.isFile()) {
				name = kj::strTree("<a href='", name.flatten(), "/contents/show'>", name.flatten(), "</a>");
			} else if(val.isFolder() || (val.isDataRef() && val.getDataRef().getDownloadStatus().isFinished())) {
				name = kj::strTree("<a href='", name.flatten(), "/show'>", name.flatten(), "</a>");
			}
			
			response = strTree(mv(response), "<tr><td>", mv(name), "</td><td>", mv(content), "</td></tr>");
		}
		
		response = strTree(mv(response), "</table>");
		
		auto flat = response.flatten();
		os.write(flat.begin(), flat.size());
	}

	//! Writes a body for a database file
	void writeFileBody(kj::BufferedOutputStream& os) {
		kj::StringPtr body = R"(<a href="contents/show">File contents</a>)";
		os.write(body.begin(), body.size());
	}
	
	//! Outputs the body of a DataRef (and links to download)
	void writeRefBody(LocalDataRef<AnyPointer> ref, const structio::Dialect& dialect, kj::BufferedOutputStream& os) {
		auto wrapped = createWrapper(ref);
		
		auto md = ref.getMetadata();
		
		auto writeStr = [&](kj::StringPtr x) {
			os.write(x.begin(), x.size());
		};
		
		if(md.getFormat().isRaw()) {
			writeStr("Raw data can not be displayed.");
			return;
		}
		
		if(!md.getFormat().isSchema()) {
			writeStr("Unknown format");
			return;
		}
		
		{
			auto backingVisitor = createVisitor(os, dialect);
			EscapingVisitor ev(*backingVisitor);
			
			structio::SaveOptions opts;
			opts.capabilityStrategy = &ev;
			
			writeStr("<pre><code>");
			structio::save(ref.getMetadata(), ev, opts);
			writeStr("</code></pre><br /><br />");
		}
		
		auto typeReader = md.getFormat().getSchema().getAs<capnp::schema::Type>();
			
		auto backingVisitor = createVisitor(os, dialect);
		EscapingVisitor ev(*backingVisitor);
		
		structio::SaveOptions opts;
		opts.capabilityStrategy = &ev;

		Maybe<capnp::Type> maybeType;
		try {
			maybeType = loader.getType(typeReader);
		} catch(kj::Exception e)
		{}

		KJ_IF_MAYBE(pType, maybeType) {
			auto& type = *pType;

			if(type.isText()) {
				writeStr(escape(ref.as<capnp::Text>().get()));
				return;
			}
			
			if(!type.isStruct()) {
				writeStr("Format can not be displayed");
				return;
			}
			
			auto schema = type.asStruct();
			capnp::DynamicStruct::Reader asStruct = wrapped.get().getAs<capnp::DynamicStruct>(schema);

			writeStr(kj::str(
				"<a href='download'>Download Archive</a> "
				"<br /><br />", schema.getUnqualifiedName(), "<br />"
				"<pre><code>"
			));
			structio::save(asStruct, ev, opts);
			writeStr("</code></pre>");
		} else {
			writeStr("Schema lookup for message type failed<br /><pre><code>");
			structio::save(typeReader, ev);
			writeStr("</code></pre>");
		}
	}
};

}

Own<kj::HttpService> createDataViewer(OneOf<DataRef<>::Client, Warehouse::Folder::Client, Warehouse::File<>::Client> root, capnp::SchemaLoader& loader) {
	return kj::heap<DataViewerImpl>(mv(root), loader);
}

}
