#pragma once

#include <assert.h>
#include <utility>
#include <exception>

namespace goldfish
{

class uncaught_exception_checker
{
public:
	uncaught_exception_checker()
		: m_count(std::uncaught_exceptions())
	{}

	bool operator()() const
	{
		return m_count != std::uncaught_exceptions();
	}
private:
	int m_count;
};

class is_work_needed
{
public:
	is_work_needed() = default;
	is_work_needed(const is_work_needed&) = delete;
	is_work_needed(is_work_needed&& rhs)
		: m_is_needed(rhs.m_is_needed)
	{
		rhs.m_is_needed = false;
	}
	is_work_needed& operator = (const is_work_needed&) = delete;
	is_work_needed& operator = (is_work_needed&& rhs)
	{
		std::swap(m_is_needed, rhs.m_is_needed);
	}

	void mark_work_done() { m_is_needed = false; }
	bool operator()() const { return m_is_needed; }
private:
	bool m_is_needed = true;
};

class assert_work_done
{
public:
	assert_work_done() = default;

#ifndef NDEBUG
	~assert_work_done()
	{
		assert(!m_work_needed() || m_uncaught_exceptions());
	}
#endif

	assert_work_done(assert_work_done&&) = default;
	assert_work_done& operator = (assert_work_done&&) = default;

	void mark_work_done()
	{
		#ifndef NDEBUG
		m_work_needed.mark_work_done();
		#endif
	}
	#ifndef NDEBUG
	bool is_work_done() const { return !m_work_needed(); }
	#endif

#ifndef NDEBUG
private:
	uncaught_exception_checker m_uncaught_exceptions;
	is_work_needed m_work_needed;
#endif
};

}