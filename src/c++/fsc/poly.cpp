#include "poly.h"

#include <kj/list.h>

namespace fsc {

namespace {

struct PolyNode {
	Vec2d x;
	uint32_t idx;
	
	PolyNode* prev = nullptr;
	PolyNode* next = nullptr;
	
	PolyNode(Vec2d x, uint32_t idx) :
		x(x), idx(idx)
	{}
};

Maybe<Vec2d> locateInTriangle(Vec2d p1, Vec2d p2, Vec2d p3, Vec2d x) {
	using Eigen::indexing::all;
	Mat2d m;
	m(all, 0) = p2 - p1;
	m(all, 1) = p3 - p1;
	
	Vec2d result = m.inverse() * (x - p1);
	
	if(result(0) <= 1e-10) return nullptr;
	if(result(1) <= 1e-10) return nullptr;
	if(result(0) + result(1) >= 1 - 1E-10) return nullptr;
	
	return result;
}

double doubleArea(Vec2d p1, Vec2d p2, Vec2d p3) {
	using Eigen::indexing::all;
	Mat2d m;
	m(all, 0) = p2 - p1;
	m(all, 1) = p3 - p1;
	return m.determinant();
}

double doubleArea(PolyNode* c) {
	return doubleArea(c -> prev -> x, c -> x, c -> next -> x);
}

}

Tensor<uint32_t, 2> triangulate(Tensor<double, 2> vertices) {
	KJ_REQUIRE(vertices.dimension(1) == 2);
	uint32_t nVerts = vertices.dimension(0);
	
	{
		// If polygon is closed, remote last vertex
		bool closed = false;
		
		Tensor<double, 0> norm = (vertices.chip(0, 0) - vertices.chip(nVerts - 1, 0)).square().sum().sqrt();
		if(norm() < 1e-10)
			closed = true;
		
		if(closed)
			--nVerts;
	}
	
	KJ_REQUIRE(nVerts >= 3);
	
	auto nodeBuilder = kj::heapArrayBuilder<PolyNode>(nVerts);
	for(auto i : kj::range(0, nVerts)) {
		nodeBuilder.add(Vec2d(vertices(i, 0), vertices(i, 1)), i);
	}
	
	auto nodes = nodeBuilder.finish();
	
	{
		auto link = [](PolyNode& n1, PolyNode& n2) {
			n1.next = &n2;
			n2.prev = &n1;
		};
		
		for(auto i : kj::range(0, nVerts - 1)) {
			link(nodes[i], nodes[i+1]);
		}
		link(nodes[nVerts - 1], nodes[0]);
	}
	
	// Calculate the total area to determine orientation
	double totalArea = 0;
	for(auto i : kj::range(1, nVerts - 1)) {
		totalArea += doubleArea(nodes[0].x, nodes[i].x, nodes[i+1].x);
	}
	
	auto resultNodes = kj::heapArrayBuilder<PolyNode*>(nVerts - 2);
	PolyNode* start = nodes.begin();
	
	// Each pass removes one triangle
	for(auto iPass : kj::range(0, nVerts - 2)) {
		PolyNode* best = nullptr;
		double bestArea = 0;
		
		PolyNode* current = start;
		while(current != start -> prev) {
			// Skip concave triangles
			// Multiply by totalArea to maintain proper sign
			double area = totalArea * doubleArea(current);
			if(area < 0)
				goto NEXT_ITERATION;
			
			// Skip all triangles containing one of the other edge points
			{
				PolyNode* pOther = current -> next -> next;
				while(pOther != current -> prev) {
					KJ_IF_MAYBE(pDontCare, locateInTriangle(current -> x, current -> prev -> x, current -> next -> x, pOther -> x)) {
						goto NEXT_ITERATION;
					}
					pOther = pOther -> next;
				}
			}
			
			if(best == nullptr || area > bestArea) {
				best = current;
				bestArea = area;
			}
			
			NEXT_ITERATION:
			current = current -> next;
		}
		
		KJ_REQUIRE(best != nullptr, iPass, nVerts, "Internal error in polygon triangulation");
		
		resultNodes.add(best);
		
		// Unlink middle node of triangle
		best -> prev -> next = best -> next;
		best -> next -> prev = best -> prev;
		
		// If start node is best, shift the start node
		if(start == best)
			start = best -> next;
	}
	
	// Build result tensor
	Tensor<uint32_t, 2> result(nVerts - 2, 3);
	for(auto i : kj::indices(resultNodes)) {
		PolyNode* ptr = resultNodes[i];
		
		result(i, 0) = ptr -> prev -> idx;
		result(i, 1) = ptr         -> idx;
		result(i, 2) = ptr -> next -> idx;
	}
	
	return result;
}

}