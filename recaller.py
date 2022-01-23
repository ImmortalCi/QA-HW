from torch.nn import functional as F

class Recaller(object):
    def train(self):

        loss = 0

        train_loader = None
        for step, (querys, positives, negatives) in enumerate(train_loader):
            self._optimizer.zero_grad()
            query_vecs = self._encoder(querys)
            positive_vecs = self._encoder(positives)
            negative_vecs = self._encoder(negatives)

            positive_scores = self._scorer(query_vecs, positive_vecs)
            negative_scores = self._scorer(query_vecs, negative_vecs)

            pair_loss = self._config.margin - positive_scores + negative_scores
            pair_loss = F.relu(pair_loss)

            pair_loss.mean().backward()
            loss += pair_loss.sum().item()
            clip_grad_norm_(self._encoder.parameters(), self._config.clip)
            self._optimizer.step()

            if step % 1000 == 0:
                logger.info(f'epoch: {epoch} training step: {step}')
        loss /= len(train_loader.dataset)

        return loss