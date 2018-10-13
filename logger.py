import math

class RunningAvgLogger:
    def __init__(self, log_path, max_step=20):
        self.log_path = log_path
        self.max_step = max_step
        self.items = {}

    def __exit__(self):
        self.flush()

    def add_item(self, name, value):
        if math.isnan(value):
            return False
        if name not in self.items:
            self.items[name] = {'val': float(value), 'count': 1}
        else:
            self.items[name]['count'] += 1
            self.items[name]['val'] = self.update_running_avg(self.items[name]['val'], float(value), self.items[name]['count'])
        return True

    def update_running_avg(self, avg, new_val, avg_count=None):
        if avg_count is None:
            avg_count = self.max_step
        if avg_count > self.max_step:
            avg_count = self.max_step
        return avg * (avg_count - 1.0) / avg_count + new_val * (1.0 / avg_count)

    def flush(self):
        writer = open(self.log_path, 'w')
        for name in self.items:
            writer.write("%s %f %d\n" % (name, self.items[name]['val'], self.items[name]['count']))
        writer.close()


if __name__ == '__main__':
    logger = RunningAvgLogger('log_test.txt')
    for i in range(100):
        logger.add_item('val1', i)
        if i % 2 == 0:
            logger.add_item('val2', i)
        if i % 5 == 0:
            logger.add_item('val3', i)
        logger.flush()