#pragma once

#include <cstdint>
#include <assert.h>
#include <exception>

namespace goldfish { namespace debug_checks
{
	struct no_check {};
	struct terminate_on_error
	{
		static void on_error() { std::terminate(); }
	};
	
	#ifndef GOLDFISH_DEFAULT_SHIP_ERROR_HANDLER
		#define GOLDFISH_DEFAULT_SHIP_ERROR_HANDLER no_check
	#endif

	#ifndef NDEBUG
	using default_error_handler = terminate_on_error;
	#else
	using default_error_handler = GOLDFISH_DEFAULT_SHIP_ERROR_HANDLER;
	#endif

	template <class error_handler> class container_base
	{
	public:
		container_base(container_base* p)
			: m_parent_address_and_bits(reinterpret_cast<uintptr_t>(p))
		{
			static_assert(alignof(container_base) % 4 == 0, "container_base need to be 4 byte aligned at least so we can use 2 bits of the address to store additional data");
			assert(parent() == p);
			assert(!is_locked());
			assert(!has_flag());
			if (p)
				p->lock();
		}
		container_base(const container_base&) = delete;
		container_base(container_base&& rhs)
			: m_parent_address_and_bits(rhs.m_parent_address_and_bits)
		{
			rhs.err_if_locked();
			rhs.m_parent_address_and_bits = 1;
			assert(rhs.is_locked());
			assert(rhs.parent() == nullptr);
		}
		container_base& operator = (const container_base&) = delete;
		container_base& operator = (container_base&&) = delete;

		void lock_parent()
		{
			if (auto p = parent())
				p->lock();
		}
		void unlock_parent()
		{
			assert(!is_locked());
			if (auto p = parent())
				p->unlock();
		}
	protected:
		void set_flag() { m_parent_address_and_bits |= 2; }
		void clear_flag() { m_parent_address_and_bits &= ~static_cast<uintptr_t>(2); }
		void err_if_flag_set() const
		{
			if (has_flag())
				error_handler::on_error();
		}
		void err_if_flag_not_set() const
		{
			if (!has_flag())
				error_handler::on_error();
		}
		void err_if_locked() const
		{
			if (is_locked())
				error_handler::on_error();
		}
		void unlock_parent_and_lock_self()
		{
			unlock_parent();
			lock();
		}
		void lock()
		{
			m_parent_address_and_bits |= 1;
		}
		container_base* parent() const
		{
			return reinterpret_cast<container_base*>(m_parent_address_and_bits & (~static_cast<uintptr_t>(3)));
		}
	private:
		bool is_locked() const { return (m_parent_address_and_bits & 1) != 0; }
		bool has_flag() const { return (m_parent_address_and_bits & 2) != 0; }

		void unlock() { m_parent_address_and_bits &= ~static_cast<uintptr_t>(1); }
		uintptr_t m_parent_address_and_bits;
	};

	template <class T> static std::true_type test_has_lock_parent(decltype(std::declval<T>().lock_parent())*) { return{}; }
	template <class T> static std::false_type test_has_lock_parent(...) { return{}; }
	template <class T> struct has_lock_parent : decltype(test_has_lock_parent<T>(nullptr)) {};

	template <class T> static std::true_type test_has_unlock_parent(decltype(std::declval<T>().unlock_parent())*) { return{}; }
	template <class T> static std::false_type test_has_unlock_parent(...) { return{}; }
	template <class T> struct has_unlock_parent : decltype(test_has_unlock_parent<T>(nullptr)) {};

	template <class T> std::enable_if_t< has_lock_parent<T>::value, void> lock_parent(T& t) { t.lock_parent(); }
	template <class T> std::enable_if_t<!has_lock_parent<T>::value, void> lock_parent(T& t) { }

	template <class T> std::enable_if_t< has_unlock_parent<T>::value, void> unlock_parent(T& t) { t.unlock_parent(); }
	template <class T> std::enable_if_t<!has_unlock_parent<T>::value, void> unlock_parent(T& t) { }
}}
