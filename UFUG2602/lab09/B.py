class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None

    def print_list(self):
        current = self.head
        while current:
            print(current.data, end=' ')
            current = current.next
        print()

    def promote(self, p_data):
        # TODO: Implement this method to promote the node with p_data to the head of the list
        newHead = self.head
        lstHead = None
        while newHead != None and newHead.data != p_data:
            lstHead = newHead
            newHead = newHead.next
        if newHead == None: return
        lstHead.next = newHead.next
        newHead.next = self.head
        self.head = newHead

# Example usage and implementation are omitted for brevity
lst = list(map(int, input().split(' ')))
sll = SinglyLinkedList()
lstNode = None
for i in lst:
    node = Node(i)
    if lstNode != None:
        lstNode.next = node
    else: sll.head = node
    lstNode = node

p_data = int(input())
sll.promote(p_data)
sll.print_list()