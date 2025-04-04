class CommNetwork:
    """
    通信网络仿真类，用于模拟代理之间的通信拓扑和统计通信开销。
    """
    def __init__(self, agent_ids):
        # 初始化邻居列表，默认为完全连通拓扑
        self.neighbors = {aid: set(agent_ids) - {aid} for aid in agent_ids}
        # 通信消息计数
        self.message_count = 0

    def set_topology(self, topology_type="fully_connected", **kwargs):
        """
        根据指定拓扑类型设置邻接关系。
        :param topology_type: "fully_connected", "ring", 等等
        """
        agent_ids = list(self.neighbors.keys())
        n = len(agent_ids)
        if topology_type == "fully_connected":
            self.neighbors = {aid: set(agent_ids) - {aid} for aid in agent_ids}
        elif topology_type == "ring":
            # 环形拓扑：每个节点连通前后两个节点
            self.neighbors = {aid: set() for aid in agent_ids}
            for i, aid in enumerate(agent_ids):
                neighbor1 = agent_ids[(i+1) % n]
                neighbor2 = agent_ids[(i-1) % n]
                self.neighbors[aid].update({neighbor1, neighbor2})
        # 可以扩展其他拓扑结构

    def send(self, sender_id, receiver_id, message=None):
        """
        模拟单播发送消息：如果接收方是发送方的邻居则增加消息计数。
        """
        if receiver_id in self.neighbors.get(sender_id, []):
            self.message_count += 1
            # 实际环境下可在此处理 message 内容；此处仅统计数量

    def broadcast(self, sender_id, message=None):
        """
        模拟广播消息给所有邻居。
        """
        if sender_id in self.neighbors:
            for recv in self.neighbors[sender_id]:
                self.send(sender_id, recv, message)
