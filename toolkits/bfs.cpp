/*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>

#include "core/graph.hpp"

void compute(Graph<Empty> * graph, VertexId root) {
  double exec_time = 0;
  exec_time -= get_time();

  VertexId * parent = graph->alloc_vertex_array<VertexId>(); // 表示从parent节点访问到这个点的
  VertexSubset * visited = graph->alloc_vertex_subset();
  VertexSubset * active_in = graph->alloc_vertex_subset(); // active_in是当前循环的bitmap, active_out是下次循环的bitmap
  VertexSubset * active_out = graph->alloc_vertex_subset(); // active和visited表达的意思是相同的吗

  visited->clear();
  visited->set_bit(root);
  active_in->clear();
  active_in->set_bit(root);
  graph->fill_vertex_array(parent, graph->vertices);// vid范围:[0, vertices-1], 将每个点的parent初始化为vertices, 表示还没找到parent
  parent[root] = root;

  VertexId active_vertices = 1;

  // src是否访问过，可以通过查看active_in, 也就是上轮循环的active_out(dst). dst是否访问过，不能查看active_in,需要查看visited, 也就是上轮循环的active_out
  for (int i_i=0;active_vertices>0;i_i++) {
    if (graph->partition_id==0) {
      printf("active(%d)>=%u\n", i_i, active_vertices);
    }
    active_out->clear();
    // sparse mode: [active src]->[master dst]
    active_vertices = graph->process_edges<VertexId,VertexId>( 
      [&](VertexId src) { // sparse_signal // master src
        graph->emit(src, src); // msg是src, src是src->dst的parent // 都从src出发了， src已经是活跃的了
      },
      [&](VertexId src, VertexId msg, VertexAdjList<Empty> outgoing_adj){ // sparse_slot // mirror src
        VertexId activated = 0; // 注意， 发送来的src肯定是not visited, 不需要像在dense_signal里那样check一下
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;
          if (parent[dst]==graph->vertices && cas(&parent[dst], graph->vertices, src)) { // dst还没被访问过， 它的parent点还没更新过
            active_out->set_bit(dst); // 为什么不在这里更新visited
            activated += 1;
          }
        }
        return activated;
      },
      // dense mode: [not_visited dst]<-[active master src]
      [&](VertexId dst, VertexAdjList<Empty> incoming_adj) { // dense_signal // dst是mirror, src->dst
        if (visited->get_bit(dst)) return;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          if (active_in->get_bit(src)) { // 只统计active的点和出边
            graph->emit(dst, src); // msg是src, src是src->dst的parent
            break;
          }
        }
      },
      [&](VertexId dst, VertexId msg) { // dense_slot
        if (cas(&parent[dst], graph->vertices, msg)) {
          active_out->set_bit(dst); // active_in, active_out是bfs算法里的入边和出边，和graph.hpp里的无关
          return 1;
        }
        return 0;
      },
      active_in, visited // activs, dense_selective
    );
    active_vertices = graph->process_vertices<VertexId>(
      [&](VertexId vtx) {
        visited->set_bit(vtx);
        return 1;
      },
      active_out
    );
    std::swap(active_in, active_out); // 在下一轮循环中， visited和active_in相同
  }

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  graph->gather_vertex_array(parent, 0);
  if (graph->partition_id==0) {
    VertexId found_vertices = 0;
    for (VertexId v_i=0;v_i<graph->vertices;v_i++) {
      if (parent[v_i] < graph->vertices) {
        found_vertices += 1;
      }
    }
    printf("found_vertices = %u\n", found_vertices);
  }

  graph->dealloc_vertex_array(parent);
  delete active_in;
  delete active_out;
  delete visited;
}

int main(int argc, char ** argv) {
  MPI_Instance mpi(&argc, &argv);

  if (argc<4) {
    printf("bfs [file] [vertices] [root]\n");
    exit(-1);
  }

  Graph<Empty> * graph;
  graph = new Graph<Empty>();
  VertexId root = std::atoi(argv[3]);
  graph->load_directed(argv[1], std::atoi(argv[2]));

  compute(graph, root);
  for (int run=0;run<5;run++) {
    compute(graph, root);
  }

  delete graph;
  return 0;
}
