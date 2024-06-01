#include "common.h"

namespace fsc {
	/** Cross-thread communication queue.
	 *
	 * A queue that can asynchronously poll messages while allowing cross-
	 * thread synchronous pushes. Supports connection-like close semantics
	 */
	template<typename T>
	struct XThreadQueue {
		//! Push an item into the queue. Returns true if the queue is open, and false if it closed.
		bool push(T&& t) const;
		
		/** Retrieve item from queue.
		 *
		 * Returns a promise that resolves once an entry is available in the queue. Only one
		 * such promise may be active at any time. Waiting promises will get cancelled if another
		 * wait is queued with pop().
		 **/
		Promise<T> pop();
		
		//! Removes all items currently in the queue.
		void clear();
		
		/** Close the queue
		 *
		 * After closing the queue, no new messages can be pushed into the queue. Objects that
		 * are pushed after closing will be silently discarded. Objects pushed into the queue
		 * before calling close() will remain in the queue and can be read with pop(). Once
		 * the queue is cleared, the promise returned by pop() will contain the exception
		 * argument to close()
		 */
		void close(const kj::Exception& e = KJ_EXCEPTION(DISCONNECTED, "Queue closed"));
		
		~XThreadQueue();
		
	private:
		struct Node {
			kj::ListLink<Node> link;
			T contents;
			
			Node(T&& t) : contents(mv(t)) {}
		};
		
		struct Shared {
			kj::List<Node, &Node::link> entries;
			Maybe<Own<kj::CrossThreadPromiseFulfiller<T>>> fulfiller;		
			Maybe<kj::Exception> isClosed;
		};
		
		kj::MutexGuarded<Shared> shared;
	};
}

// Inline implementation

namespace fsc {
	template<typename T>
	bool XThreadQueue<T>::push(T&& t) const {
		auto locked = shared.lockExclusive();
		
		if(locked -> isClosed != nullptr) {
			return false;
		}
		
		KJ_IF_MAYBE(pFulfiller, locked -> fulfiller) {
			(**pFulfiller).fulfill(kj::mv(t));
		} else {
			auto node = new Node(mv(t));
			locked -> entries.add(*node);
		}
		
		locked -> fulfiller = nullptr;
		
		return true;
	}

	template<typename T>
	Promise<T> XThreadQueue<T>::pop() {
		auto locked = shared.lockExclusive();
		
		if(locked -> entries.size() > 0) {
			auto node = &(locked -> entries.front());
			Promise<T> result = mv(node -> contents);
			locked -> entries.remove(*node);
			delete node;
			return result;
		}
		
		KJ_IF_MAYBE(pExc, locked -> isClosed) {
			//kj::throwFatalException(cp(*pExc));
			return cp(*pExc);
		}
		
		auto paf = kj::newPromiseAndCrossThreadFulfiller<T>();
		locked -> fulfiller = mv(paf.fulfiller);
		return mv(paf.promise);
	}
	
	template<typename T>
	void XThreadQueue<T>::clear() {
		auto locked = shared.lockExclusive();
		auto& e = locked -> entries;
		
		while(e.size() != 0) {
			auto front = &(e.front());
			e.remove(*front);
			delete front;
		}
	}

	template<typename T>
	void XThreadQueue<T>::close(const kj::Exception& reason) {
		auto locked = shared.lockExclusive();
		auto& e = locked -> entries;
		
		KJ_IF_MAYBE(pFulfiller, locked -> fulfiller) {
			(**pFulfiller).reject(cp(reason));
		}
		locked -> fulfiller = nullptr;
		locked -> isClosed = reason;
	}

	template<typename T>
	XThreadQueue<T>::~XThreadQueue() {
		clear();
		close();
	}

}