namespace fscpy {

template<typename T>
struct UnknownHolder : public UnknownObject {
	static inline const int typeSlot = 0;
	T val;
	
	template<typename T2>
	UnknownHolder(T2&& t) : val(fwd<T2>(t)) {
		type = &typeSlot;
	}
	~UnknownHolder () noexcept {}
};

template<typename T>
UnknownObject* eraseType(T&& t) { return new UnknownHolder<kj::Decay<T>>(fwd<T>(t)); }
	
template<typename T>
Maybe<T&> checkType(UnknownObject& o) {
	if(o.type != &UnknownHolder<T>::typeSlot)
		return nullptr;
	
	auto holder = static_cast<UnknownHolder<T>&>(o);
	return holder.val;
}

template<typename T>
py::object unknownObject(T&& ref) {
	return py::cast(eraseType(fwd<T>(ref)));
}
		
template<typename T, typename ShapeContainer>
ContiguousCArray ContiguousCArray::alloc(ShapeContainer& requestedShape, kj::StringPtr formatCode) {
	ContiguousCArray result;
	
	size_t shapeProd = 1;
	
	result.shape.resize(requestedShape.size());
	for(auto i : kj::indices(requestedShape)) {
		result.shape[i] = requestedShape[i];
		shapeProd *= result.shape[i];
	}
	
	result.elementSize = sizeof(T);
	result.data = kj::heapArray<unsigned char>(shapeProd * sizeof(T));
	result.format = kj::heapString(formatCode);
	
	return result;
}

template<typename T>
kj::ArrayPtr<T> ContiguousCArray::as() {
	return kj::ArrayPtr<T>((T*) data.begin(), data.size() / sizeof(T));
}

}