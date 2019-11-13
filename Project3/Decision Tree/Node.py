class Node:
    def __init__(self, attribute_index, attribute_value):
        self.attribute_index = attribute_index
        self.attribute_value = attribute_value
        self.left = None
        self.right = None
        self.leaf_node = False
        self.prediction = None