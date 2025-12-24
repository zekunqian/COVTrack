import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from mmdet.models import LOSSES
import torch.nn.functional as F

@LOSSES.register_module(force=True)
class CycleLoss(nn.Module):
    def __init__(self, margin=0.5, loss_type=['pairwise', 'triplewise']):
        super(CycleLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.delta = 0.5
        self.m = margin
        self.epsilon = 0.1
        self.loss_type=loss_type

    def pairwise_loss(self, all_S):
        loss_num = 0
        loss_sum = 0
        for i in range(len(all_S)):
            for j in range(len(all_S)):
                if i < j:
                    loss_num += 1
                    S = all_S[i][j]
                    if S.shape[0] < S.shape[1]:
                        S21 = S
                        S12 = S21.transpose(1, 0)
                    else:
                        S12 = S
                        S21 = S12.transpose(1, 0)

                    scale12 = np.log(self.delta / (1 - self.delta) * max(S12.size(1), 1)) / self.epsilon
                    scale21 = np.log(self.delta / (1 - self.delta) * max(S21.size(1), 1)) / self.epsilon
                    S12_hat = f.softmax(S12 * scale12, dim=1)
                    S21_hat = f.softmax(S21 * scale21, dim=1)
                    S1221_hat = torch.mm(S12_hat, S21_hat)
                    n = S1221_hat.shape[0]
                    I = torch.eye(n).cuda()
                    # I = torch.eye(n)
                    pos = S1221_hat * I
                    neg = S1221_hat * (1 - I)
                    loss = 0
                    loss += torch.sum(f.relu(torch.max(neg, 1)[0] + self.m - torch.diag(pos)))
                    loss += torch.sum(f.relu(torch.max(neg, 0)[0] + self.m - torch.diag(pos)))
                    loss /= 2 * n
                    loss_sum += loss
        return loss_sum / loss_num

    def triplewise_loss(self, all_S):
        loss_num = 0
        loss_sum = 0
        for i in range(len(all_S)):
            for j in range(len(all_S)):
                if i < j:
                    for k in range(len(all_S)):
                        if k != i and k != j :
                            loss_num += 1
                            S12_ = all_S[i][k]
                            S23_ = all_S[k][j]
                            S = torch.mm(S12_, S23_)
                            if S.shape[0] < S.shape[1]:
                                S21 = S
                                S12 = S21.transpose(1, 0)
                            else:
                                S12 = S
                                S21 = S12.transpose(1, 0)
                            scale12 = np.log(self.delta / (1 - self.delta) * max(S12.size(1),1)) / self.epsilon
                            scale21 = np.log(self.delta / (1 - self.delta) * max(S21.size(1),1)) / self.epsilon
                            S12_hat = f.softmax(S12 * scale12, dim=1)
                            S21_hat = f.softmax(S21 * scale21, dim=1)
                            S1221_hat = torch.mm(S12_hat, S21_hat)
                            n = S1221_hat.shape[0]
                            I = torch.eye(n).cuda()
                            pos = S1221_hat * I
                            neg = S1221_hat * (1 - I)
                            loss = 0
                            loss += torch.sum(f.relu(torch.max(neg, 1)[0] + self.m - torch.diag(pos)))
                            loss += torch.sum(f.relu(torch.max(neg, 0)[0] + self.m - torch.diag(pos)))
                            loss /= 2 * n
                            loss_sum += loss
        return loss_sum / loss_num

    def gen_X_S(self, feature_ls: list):
        norm_feature = [f.normalize(i, dim=-1) for i in feature_ls]
        all_blocks_S = []
        all_blocks_X = []
        for idx, x in enumerate(norm_feature):
            row_blocks_S = []
            row_blocks_X = []
            for idy, y in enumerate(norm_feature):
                S = torch.mm(x, y.transpose(0, 1))
                scale = np.log(self.delta / (1 - self.delta) * max(S.size(1),1)) / self.epsilon
                S_hat = f.softmax(S * scale, dim=1)
                row_blocks_X.append(S_hat)
                row_blocks_S.append(S)
            row_blocks_X = torch.cat(row_blocks_X, dim=1)
            all_blocks_S.append(row_blocks_S)
            all_blocks_X.append(row_blocks_X)
        all_blocks_X = torch.cat(all_blocks_X, dim=0)
        return all_blocks_S, all_blocks_X
    def get_pair_consistency(self, feats_list):
        assert len(feats_list) == 2, f"The size of feats_list is not 2, as length is {len(feats_list)}! "
        norm_feature_list = [f.normalize(i, dim=-1) for i in feats_list]
        x = norm_feature_list[0]
        y = norm_feature_list[1]
        S = torch.mm(x, y.transpose(0, 1))
        # scale = np.log(self.delta / (1 - self.delta) * max(S.size(1), 1)) / self.epsilon
        # S_hat = f.softmax(S * scale, dim=1)
        S12 = S
        S21 = S12.transpose(1, 0)
        scale12 = np.log(self.delta / (1 - self.delta) * max(S12.size(1), 1)) / self.epsilon
        scale21 = np.log(self.delta / (1 - self.delta) * max(S21.size(1), 1)) / self.epsilon
        S12_hat = f.softmax(S12 * scale12, dim=1)
        S21_hat = f.softmax(S21 * scale21, dim=1)
        S1221_hat = torch.mm(S12_hat, S21_hat)
        return S1221_hat, torch.diag(S1221_hat)
    def get_triple_consistency(self, feats_list):
        assert len(feats_list) == 3, f"The size of feats_list is not 3, as length is {len(feats_list)}!"
        norm_feature_list = [f.normalize(i, dim=-1) for i in feats_list]
        x = norm_feature_list[0]
        y = norm_feature_list[1]
        z = norm_feature_list[2]
        S13 = torch.mm(x, z.transpose(0, 1))
        S32 = torch.mm(z, y.transpose(0, 1))
        S = torch.mm(S13, S32)
        S12 = S
        S21 = S12.transpose(1, 0)
        scale12 = np.log(self.delta / (1 - self.delta) * max(S12.size(1), 1)) / self.epsilon
        scale21 = np.log(self.delta / (1 - self.delta) * max(S21.size(1), 1)) / self.epsilon
        S12_hat = f.softmax(S12 * scale12, dim=1)
        S21_hat = f.softmax(S21 * scale21, dim=1)
        S1221_hat = torch.mm(S12_hat, S21_hat)
        return S1221_hat, torch.diag(S1221_hat)

    def forward(self, feature_ls):
        S, X = self.gen_X_S(feature_ls)
        tmp = self.get_pair_consistency(feature_ls[:2])
        loss = 0
        if 'pairwise' in self.loss_type:
            loss += self.pairwise_loss(S)
        if 'triplewise' in self.loss_type:
            loss += self.triplewise_loss(S)
        return loss


@LOSSES.register_module(force=True)
class ComprehensiveMatchingSupervision:
    def __init__(self, feature_dim=256):
        """
        初始化综合匹配监督
        Args:
            feature_dim: 目标特征维度，默认256
        """
        self.feature_dim = feature_dim
        self.delta = 0.5  # softmax缩放因子
        self.epsilon = 0.1  # 数值稳定性参数

    def compute_similarity(self, feat1, feat2):
        """
        计算两组特征之间的相似度矩阵
        Args:
            feat1: 图1的目标特征 (n1, 256)
            feat2: 图2的目标特征 (n2, 256)
        Returns:
            similarity: 相似度矩阵 (n1, n2)
        """
        # 归一化特征
        feat1_norm = F.normalize(feat1, p=2, dim=1)
        feat2_norm = F.normalize(feat2, p=2, dim=1)

        # 计算相似度矩阵
        similarity = torch.mm(feat1_norm, feat2_norm.t())
        return similarity

    def compute_scaled_softmax(self, S):
        """
        计算带缩放的softmax概率
        Args:
            S: 相似度矩阵
        Returns:
            S12_prob, S21_prob: 缩放后的softmax概率
        """
        # if S.shape[0] < S.shape[1]:
        #     S21 = S
        #     S12 = S21.transpose(1, 0)
        # else:
        S12 = S
        S21 = S12.transpose(1, 0)

            # 计算缩放因子
        scale12 = np.log(self.delta / (1 - self.delta) * max(S12.size(1), 1)) / self.epsilon
        scale21 = np.log(self.delta / (1 - self.delta) * max(S21.size(1), 1)) / self.epsilon

        # 应用softmax
        S12_prob = F.softmax(S12 * scale12, dim=1)
        S21_prob = F.softmax(S21 * scale21, dim=1)

        return S12_prob, S21_prob

    def compute_loss(self, feat1, feat2, assignment_matrix):
        """
        计算综合匹配损失
        Args:
            feat1: 图1的目标特征 (n1, 256)
            feat2: 图2的目标特征 (n2, 256)
            assignment_matrix: 两图之间的指派矩阵 (n1, n2)，1表示匹配，0表示不匹配
        """
        # 计算相似度矩阵
        S = self.compute_similarity(feat1, feat2)  # (n1, n2)

        # 使用缩放的softmax获得概率分布
        S12_prob, S21_prob = self.compute_scaled_softmax(S)

        # 1. 路径一致性损失
        path_loss = self.compute_path_loss(S12_prob, S21_prob, assignment_matrix)

        # 2. 唯一性损失
        uniqueness_loss = self.compute_uniqueness_loss(S12_prob, S21_prob)

        # 3. 循环一致性损失
        consistency_loss = self.compute_consistency_loss(S12_prob, S21_prob, assignment_matrix)

        # 4. 特征相似度损失
        # similarity_loss = self.compute_similarity_loss(S, assignment_matrix)

        # total_loss = path_loss + 0.1 * uniqueness_loss + 0.1 * consistency_loss + 0.1 * similarity_loss
        total_loss = path_loss + 0.1 * uniqueness_loss + 0.1 * consistency_loss

        return {
            'total_loss': total_loss,
            'path_loss': path_loss,
            'uniqueness_loss': uniqueness_loss,
            'consistency_loss': consistency_loss,
            # 'similarity_loss': similarity_loss
        }

    def compute_path_loss(self, S12_prob, S21_prob, assignment_matrix):
        """计算路径一致性损失"""
        # 对匹配对计算损失
        positive_mask = (assignment_matrix == 1)

        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=S12_prob.device)

            # 正确路径的概率
        correct_paths = S12_prob[positive_mask] * S21_prob.t()[positive_mask]
        path_loss = -torch.log(correct_paths + 1e-6).mean()

        return path_loss

    def compute_uniqueness_loss(self, S12_prob, S21_prob):
        """计算唯一性损失（使用熵）"""
        # 计算行和列的熵
        row_entropy = -(S12_prob * torch.log(S12_prob + 1e-6)).sum(dim=1).mean()
        col_entropy = -(S21_prob * torch.log(S21_prob + 1e-6)).sum(dim=1).mean()

        return row_entropy + col_entropy

    def compute_consistency_loss(self, S12_prob, S21_prob, assignment_matrix):
        """计算循环一致性损失"""
        # 计算循环一致性矩阵
        S1221_prob = torch.mm(S12_prob, S21_prob)

        # 对角线应该接近assignment_matrix指定的值
        diagonal_mask = torch.eye(S1221_prob.shape[0], device=S1221_prob.device)
        positive_mask = (assignment_matrix.sum(dim=1) > 0).float()

        # 计算对角线损失
        diagonal_loss = -torch.log(torch.clamp(S1221_prob * diagonal_mask * positive_mask.unsqueeze(1), min=1e-6)).sum()

        # 非对角线应该接近0
        off_diagonal_loss = (S1221_prob * (1 - diagonal_mask)).sum()

        return (diagonal_loss + 0.1 * off_diagonal_loss) / S1221_prob.shape[0]

    def compute_similarity_loss(self, S12_hat, assignment_matrix):
        """计算特征相似度损失"""
        # deprecated methods
        # 正样本对的相似度应该大，负样本对的相似度应该小
        positive_mask = (assignment_matrix == 1)
        negative_mask = (assignment_matrix == 0)

        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=S12_hat.device)

            # 计算对比损失
        positive_sim = S12_hat[positive_mask]
        negative_sim = S12_hat[negative_mask]

        # 使用margin-based损失
        margin = 0.3
        loss = torch.clamp(margin - positive_sim, min=0).mean() + \
               torch.clamp(negative_sim - margin, min=0).mean()

        return loss



if __name__ == '__main__':
    feature_ls = torch.load('/home/clark/test/test.pt')
    cycleLoss = CycleLoss()
    t1, t1_diag = cycleLoss.get_pair_consistency(feature_ls[:2])
    t2, t2_diag = cycleLoss.get_triple_consistency(feature_ls[:3])
    print('hello world')

    # test new loss
    supervisor = ComprehensiveMatchingSupervision()


    def create_assignment_matrix(n1=44, n2=37):
        # 创建全零矩阵
        assignment_matrix = torch.zeros(n1, n2)

        # 随机选择一些位置设置为1
        # 确保每行最多只有一个1（一个源目标最多匹配一个目标）
        num_matches = min(n1, n2)  # 匹配数不能超过较小的维度
        num_actual_matches = int(num_matches * 0.7)  # 假设70%的目标有匹配

        # 随机选择行和列的索引
        rows = torch.randperm(n1)[:num_actual_matches]
        cols = torch.randperm(n2)[:num_actual_matches]

        # 设置匹配关系
        for i in range(num_actual_matches):
            assignment_matrix[rows[i], cols[i]] = 1

        return assignment_matrix

        # 创建指派矩阵


    assignment_matrix = create_assignment_matrix(feature_ls[0].shape[0], feature_ls[1].shape[0])
    losses = supervisor.compute_loss(feature_ls[0], feature_ls[1], assignment_matrix)