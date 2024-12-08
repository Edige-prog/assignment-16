from typing import List, Any, Dict, Set, Generator

class StaticArray:
    def __init__(self, capacity: int):
        """
        Initialize a static array of a given capacity.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity = capacity
        self.array = [None] * capacity

    def set(self, index: int, value: int) -> None:
        """
        Set the value at a particular index.
        """
        if not 0 <= index < self.capacity:
            raise IndexError("Index out of bounds.")
        self.array[index] = value

    def get(self, index: int) -> int:
        """
        Retrieve the value at a particular index.
        """
        if not 0 <= index < self.capacity:
            raise IndexError("Index out of bounds.")
        return self.array[index]

class DynamicArray:
    def __init__(self):
        """
        Initialize an empty dynamic array.
        """
        self.capacity = 2  # Initial capacity
        self.size = 0
        self.array = [None] * self.capacity

    def _resize(self, new_capacity):
        """
        Resize the internal array to a new capacity.
        """
        new_array = [None] * new_capacity
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array
        self.capacity = new_capacity

    def append(self, value: int) -> None:
        """
        Add a value to the end of the dynamic array.
        """
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        self.array[self.size] = value
        self.size += 1

    def insert(self, index: int, value: int) -> None:
        """
        Insert a value at a particular index.
        """
        if not 0 <= index <= self.size:
            raise IndexError("Index out of bounds.")
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        for i in range(self.size, index, -1):
            self.array[i] = self.array[i - 1]
        self.array[index] = value
        self.size += 1

    def delete(self, index: int) -> None:
        """
        Delete the value at a particular index.
        """
        if not 0 <= index < self.size:
            raise IndexError("Index out of bounds.")
        for i in range(index, self.size - 1):
            self.array[i] = self.array[i + 1]
        self.array[self.size - 1] = None
        self.size -= 1
        if 0 < self.size < self.capacity // 4:
            self._resize(self.capacity // 2)

    def get(self, index: int) -> int:
        """
        Retrieve the value at a particular index.
        """
        if not 0 <= index < self.size:
            raise IndexError("Index out of bounds.")
        return self.array[index]



class Node:
    def __init__(self, value: int):
        """
        Initialize a node.
        """
        self.value = value
        self.next = None


class SinglyLinkedList:
    def __init__(self):
        """
        Initialize an empty singly linked list.
        """
        self.head = None
        self._size = 0

    def append(self, value: int) -> None:
        """
        Add a node with a value to the end of the linked list.
        """
        new_node = Node(value)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self._size += 1

    def insert(self, position: int, value: int) -> None:
        """
        Insert a node with a value at a particular position.
        """
        if position < 0 or position > self._size:
            raise IndexError("Position out of bounds.")
        new_node = Node(value)
        if position == 0:
            new_node.next = self.head
            self.head = new_node
        else:
            prev = self.head
            for _ in range(position - 1):
                prev = prev.next
            new_node.next = prev.next
            prev.next = new_node
        self._size += 1

    def delete(self, value: int) -> None:
        """
        Delete the first node with a specific value.
        """
        current = self.head
        prev = None
        while current:
            if current.value == value:
                if prev:
                    prev.next = current.next
                else:
                    self.head = current.next
                self._size -= 1
                return
            prev = current
            current = current.next
        raise ValueError(f"Value {value} not found in the list.")

    def find(self, value: int) -> Node:
        """
        Find a node with a specific value.
        """
        current = self.head
        while current:
            if current.value == value:
                return current
            current = current.next
        raise ValueError(f"Value {value} not found in the list.")

    def size(self) -> int:
        """
        Returns the number of elements in the linked list.
        """
        return self._size

    def is_empty(self) -> bool:
        """
        Checks if the linked list is empty.
        """
        return self._size == 0

    def print_list(self) -> None:
        """
        Prints all elements in the linked list.
        """
        elements = []
        current = self.head
        while current:
            elements.append(str(current.value))
            current = current.next
        print(" -> ".join(elements))

    def reverse(self) -> None:
        """
        Reverse the linked list in-place.
        """
        prev = None
        current = self.head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev

    def get_head(self) -> Node:
        """
        Returns the head node of the linked list.
        """
        return self.head

    def get_tail(self) -> Node:
        """
        Returns the tail node of the linked list.
        """
        if not self.head:
            return None
        current = self.head
        while current.next:
            current = current.next
        return current

class DoubleNode:
    def __init__(self, value: int, next_node=None, prev_node=None):
        """
        Initialize a double node with value, next, and previous.
        """
        self.value = value
        self.next = next_node
        self.prev = prev_node


class DoublyLinkedList:
    def __init__(self):
        """
        Initialize an empty doubly linked list.
        """
        self.head = None
        self.tail = None
        self._size = 0

    def append(self, value: int) -> None:
        """
        Add a node with a value to the end of the linked list.
        """
        new_node = DoubleNode(value)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self._size += 1

    def insert(self, position: int, value: int) -> None:
        """
        Insert a node with a value at a particular position.
        """
        if position < 0 or position > self._size:
            raise IndexError("Position out of bounds.")
        new_node = DoubleNode(value)
        if position == 0:
            if not self.head:
                self.head = self.tail = new_node
            else:
                new_node.next = self.head
                self.head.prev = new_node
                self.head = new_node
        elif position == self._size:
            self.append(value)
            return
        else:
            current = self.head
            for _ in range(position):
                current = current.next
            prev_node = current.prev
            new_node.next = current
            new_node.prev = prev_node
            prev_node.next = new_node
            current.prev = new_node
        self._size += 1

    def delete(self, value: int) -> None:
        """
        Delete the first node with a specific value.
        """
        current = self.head
        while current:
            if current.value == value:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev
                self._size -= 1
                return
            current = current.next
        raise ValueError(f"Value {value} not found in the list.")

    def find(self, value: int) -> DoubleNode:
        """
        Find a node with a specific value.
        """
        current = self.head
        while current:
            if current.value == value:
                return current
            current = current.next
        raise ValueError(f"Value {value} not found in the list.")

    def size(self) -> int:
        """
        Returns the number of elements in the linked list.
        """
        return self._size

    def is_empty(self) -> bool:
        """
        Checks if the linked list is empty.
        """
        return self._size == 0

    def print_list(self) -> None:
        """
        Prints all elements in the linked list.
        """
        elements = []
        current = self.head
        while current:
            elements.append(str(current.value))
            current = current.next
        print(" <-> ".join(elements))

    def reverse(self) -> None:
        """
        Reverse the linked list in-place.
        """
        current = self.head
        self.head, self.tail = self.tail, self.head
        while current:
            current.prev, current.next = current.next, current.prev
            current = current.prev  # because we swapped next and prev

    def get_head(self) -> DoubleNode:
        """
        Returns the head node of the linked list.
        """
        return self.head

    def get_tail(self) -> DoubleNode:
        """
        Returns the tail node of the linked list.
        """
        return self.tail

class Queue:
    def __init__(self):
        """
        Initialize an empty queue.
        """
        self.items = []

    def enqueue(self, value: int) -> None:
        """
        Add a value to the end of the queue.
        """
        self.items.append(value)

    def dequeue(self) -> int:
        """
        Remove a value from the front of the queue and return it.
        """
        if self.is_empty():
            raise IndexError("Dequeue from empty queue.")
        return self.items.pop(0)

    def peek(self) -> int:
        """
        Peek at the value at the front of the queue without removing it.
        """
        if self.is_empty():
            raise IndexError("Peek from empty queue.")
        return self.items[0]

    def is_empty(self) -> bool:
        """
        Check if the queue is empty.
        """
        return len(self.items) == 0


class TreeNode:
    def __init__(self, value: int):
        """
        Initialize a tree node with value.
        """
        self.value = value
        self.left = None
        self.right = None


class BinarySearchTree:
    def __init__(self):
        """
        Initialize an empty binary search tree.
        """
        self.root = None
        self._size = 0

    def insert(self, value: int) -> None:
        """
        Insert a node with a specific value into the binary search tree.
        """
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)
        self._size += 1

    def _insert_recursive(self, node: TreeNode, value: int) -> None:
        if value < node.value:
            if node.left:
                self._insert_recursive(node.left, value)
            else:
                node.left = TreeNode(value)
        elif value > node.value:
            if node.right:
                self._insert_recursive(node.right, value)
            else:
                node.right = TreeNode(value)
        else:
            raise ValueError("Duplicate values are not allowed in BST.")

    def delete(self, value: int) -> None:
        """
        Remove a node with a specific value from the binary search tree.
        """
        self.root, deleted = self._delete_recursive(self.root, value)
        if deleted:
            self._size -= 1
        else:
            return None

    def _delete_recursive(self, node: TreeNode, value: int):
        if not node:
            return node, False
        deleted = False
        if value < node.value:
            node.left, deleted = self._delete_recursive(node.left, value)
        elif value > node.value:
            node.right, deleted = self._delete_recursive(node.right, value)
        else:
            deleted = True
            if not node.left and not node.right:
                return None, True
            elif not node.left:
                return node.right, True
            elif not node.right:
                return node.left, True
            else:
                # Find inorder successor
                successor = self._minimum_node(node.right)
                node.value = successor.value
                node.right, _ = self._delete_recursive(node.right, successor.value)
        return node, deleted

    def search(self, value: int) -> TreeNode:
        """
        Search for a node with a specific value in the binary search tree.
        """
        return self._search_recursive(self.root, value)

    def _search_recursive(self, node: TreeNode, value: int) -> TreeNode:
        if not node:
            return None
        if value < node.value:
            return self._search_recursive(node.left, value)
        elif value > node.value:
            return self._search_recursive(node.right, value)
        else:
            return node

    def inorder_traversal(self) -> List[int]:
        """
        Perform an in-order traversal of the binary search tree.
        """
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node: TreeNode, result: List[int]) -> None:
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

    def preorder_traversal(self) -> List[int]:
        """
        Perform a pre-order traversal of the tree.
        """
        result = []
        self._preorder_recursive(self.root, result)
        return result

    def _preorder_recursive(self, node: TreeNode, result: List[int]) -> None:
        if node:
            result.append(node.value)
            self._preorder_recursive(node.left, result)
            self._preorder_recursive(node.right, result)

    def postorder_traversal(self) -> List[int]:
        """
        Perform a post-order traversal of the tree.
        """
        result = []
        self._postorder_recursive(self.root, result)
        return result

    def _postorder_recursive(self, node: TreeNode, result: List[int]) -> None:
        if node:
            self._postorder_recursive(node.left, result)
            self._postorder_recursive(node.right, result)
            result.append(node.value)

    def level_order_traversal(self) -> List[int]:
        """
        Perform a level order (breadth-first) traversal of the tree.
        """
        result = []
        if not self.root:
            return result
        queue = [self.root]
        while queue:
            current = queue.pop(0)
            result.append(current.value)
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
        return result

    def minimum(self) -> TreeNode:
        """
        Returns the node with the minimum value in the tree.
        """
        if not self.root:
            raise ValueError("BST is empty.")
        return self._minimum_node(self.root)

    def _minimum_node(self, node: TreeNode) -> TreeNode:
        current = node
        while current.left:
            current = current.left
        return current

    def maximum(self) -> TreeNode:
        """
        Returns the node with the maximum value in the tree.
        """
        if not self.root:
            raise ValueError("BST is empty.")
        return self._maximum_node(self.root)

    def _maximum_node(self, node: TreeNode) -> TreeNode:
        current = node
        while current.right:
            current = current.right
        return current

    def size(self) -> int:
        """
        Returns the number of nodes in the tree.
        """
        return self._size

    def is_empty(self) -> bool:
        """
        Checks if the tree is empty.
        """
        return self._size == 0

    def height(self) -> int:
        """
        Returns the height of the tree.
        """
        return self._height_recursive(self.root)

    def _height_recursive(self, node: TreeNode) -> int:
        if not node:
            return 0  # Height of empty tree is -1
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        return 1 + max(left_height, right_height)

    def is_valid_bst(self) -> bool:
        """
        Check if the tree is a valid binary search tree.
        """
        return self._is_valid_bst_recursive(self.root, float('-inf'), float('inf'))

    def _is_valid_bst_recursive(self, node: TreeNode, min_val: int, max_val: int) -> bool:
        if not node:
            return True
        if not (min_val < node.value < max_val):
            return False
        return (self._is_valid_bst_recursive(node.left, min_val, node.value) and
                self._is_valid_bst_recursive(node.right, node.value, max_val))



def insertion_sort(lst: List[int]) -> List[int]:
    """
    Sorts the list using insertion sort algorithm.
    """
    for i in range(1, len(lst)):
        key = lst[i]
        j = i - 1
        while j >= 0 and lst[j] > key:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = key
    return lst

def selection_sort(lst: List[int]) -> List[int]:
    """
    Sorts the list using selection sort algorithm.
    """
    n = len(lst)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if lst[j] < lst[min_idx]:
                min_idx = j
        # Swap the found minimum element with the first element
        lst[i], lst[min_idx] = lst[min_idx], lst[i]
    return lst

def bubble_sort(lst: List[int]) -> List[int]:
    """
    Sorts the list using bubble sort algorithm.
    """
    n = len(lst)
    for i in range(n):
        swapped = False
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if lst[j] > lst[j + 1]:
                # Swap if elements are in wrong order
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                swapped = True
        if not swapped:
            # Stop if the list is already sorted
            break
    return lst

def shell_sort(lst: List[int]) -> List[int]:
    """
    Sorts the list using shell sort algorithm.
    """
    n = len(lst)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = lst[i]
            j = i
            while j >= gap and lst[j - gap] > temp:
                lst[j] = lst[j - gap]
                j -= gap
            lst[j] = temp
        gap //= 2
    return lst

def merge_sort(lst: List[int]) -> List[int]:
    """
    Sorts the list using merge sort algorithm.
    """
    if len(lst) <= 1:
        return lst

    def merge(left, right):
        result = []
        i = j = 0
        # Merge the two halves
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        # Append remaining elements
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    mid = len(lst) // 2
    left = merge_sort(lst[:mid])
    right = merge_sort(lst[mid:])
    return merge(left, right)

def quick_sort(lst: List[int]) -> List[int]:
    """
    Sorts the list using quick sort algorithm.
    """
    if len(lst) <= 1:
        return lst
    pivot = lst[len(lst) // 2]
    left = [x for x in lst if x < pivot]
    middle = [x for x in lst if x == pivot]
    right = [x for x in lst if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
