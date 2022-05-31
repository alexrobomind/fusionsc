#pragma once
 
 #ifndef DOXYGEN
 #error "This file is not meant to be included"
 #endif
 
 #include "common.h"
 
/**
 * \file
 *
 * This file contains documentation for external and generated classes. It contains
 * simplified interfaces and is not intended for inclusion into user or library code.
 */
 
/**
 * \brief Cap'n'Proto library (https;//capnproto.org)
 */
 namespace capnp {
	
	/**
	 * This class forms the base class for all remotely callable objects. It represents
	 * an interface that can be exchanged with remote peers inside messages. Instances of
	 * such an interface are called capabilities.
	 *
	 * \headerfile capnp/capability.h
	 */
	struct Capability {
		/**
		 * Access point to use the functionalities provided by an interface. Can hold a
		 * local or a remote capability. These objects can be put into Cap'n'proto messages
		 * at appropriate points. When the messages are exchanged, the exchanged capabilities
		 * are tracked by both ends of the connection, and the receiver will create proxies
		 * for newly received capabilities.
		 */
		struct Client {
			//! Clients are copyable (via refcounting)
			Client(Client&);
			
			//! Clients can be created from promises
			Client(kj::Promise<Client>);
			
			//! Clients can be created from Server s
			Client(kj::Own<Server>);
		};
		
		/**
		 * A backend that provides the functionality provided by an interface. Instances of
		 * the capability are created from instances of this class (more specifically, usually
		 * a subclass) and converted into local clients which are then passed around.
		 */
		struct Server {
		};
	};
 }
 
/**
 * \brief KJ library (https;//capnproto.org)
 */
 namespace kj {
	 //! Promise for an eventual instance of T
	 /**
	  * \headerfile kj/async.h ""
	  */
	 template<typename T>
	 struct Promise {
	 };
	 
	 //! Ownership-holding smart pointer (similar to std::unique_ptr)
	 /**
	  * An owning pointer that can be moved but not copied. Unlike std::unique_tr,
	  * this class supports inheritance, particularly the usage of virtual destructors
	  */
	  template<typename T>
	  struct Own {};
 }
 
 namespace fsc{
	/** 
	 *  \tparam T Type of the root message stored in the data ref.
	 *
	 *  The DataRef template is a special capability recognized all throughout the FSC library.
	 *  It represents a link to a data storage location (local or remote), associated with abort
	 *  unique ID, which can be downloaded to local storage and accessed there. Locally downloaded
	 *  data are represented by the LocalDataRef class, which subclasses DataRef::Client.
	 *
	 *  The stored data is expected to be in one of the two following formats:
	 *  - If T is capnp::Data, then the dataref stores a raw binary data array
	 *  - If T is any other type, it must correspond to a Cap'n'proto struct type. In
	 *    this case, the DataRef holds a message with the corresponding root type, and
	 *    a capability table that tracks any remote objects in use by this message
	 *    (including other DataRef instances).
	 *
	 *  Once obtained, DataRefs can be freely passed around as part of RPC calls or data published
	 *  in other DataRef::Client instances. The fsc runtime will do all it possibly can to protect the integrity
	 *  of a DataRef. In the absence of hardware failure, data referenced via DataRef objects
	 *  can only go out of use once all referencing DataRef objects do so as well.
	 *
	 *  DataRefs only represent a link to locally or remotely stored data. To access the underlying
	 *  data, they must be converted into LocalDataRef instances using LocalDataService::download() methods.
	 */
	template<typename T>
	struct DataRef : public capnp::Capability {
		//! DataRef client
		struct Client : public virtual capnp::Capability::Client {
		};
	};
 }
 
 //! The Eigen library (https://eigen.tuxfamily.org)
 namespace Eigen {
	 //! Vector class
	 template<typename T, int dim>
	 struct Vector {};
	 //! Matrix class
	 template<typename T, int m, int n>
	 struct Vector {};
	 
	 //! Tensor of fixed size and dimensionality
	 template<typename T, typename Sizes>
	 struct TensorFixedSize {};
	 
	 //! Reference to a tensor expression, which will evaluate lazily
	 template<typename T>
	 struct TensorRef {};
	 
	 //! Reference to an externally allocated tensor
	 template<typename T>
	 struct TensorMap {};
	 
	 //! Cost descriptor for individual kernel invocation
	 /**
	  * \ingroup kernelAPI
	  */
	 struct TensorOpCost {};
 }