# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:48:43 2024

@author: appari neeraj
"""

# BFS
graph= {
'5' : ['3','7'],
'3' : ['2','4'],
'7' : ['8'],
'2' : [],
'4' : ['8'],
'8' : []
}
visited = []
queue = []
def bfs(visited,graph,node):
    visited.append(node)
    queue.append(node)
    while queue:
        m = queue.pop(0)
        print(m,end = "")
    for neighbour in graph[m]:

        if neighbour not in visited:
            visited.append(neighbour)
            queue.append(neighbour)

print("following is the Breadth First Search : ")
bfs(visited,graph,'7')