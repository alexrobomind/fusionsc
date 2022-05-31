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
 *
 * \mainpage Documentation
 * \page Networking and data representation
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