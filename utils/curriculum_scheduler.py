class CurriculumScheduler:
    def __init__(self, stages):
        """
        初始化课程学习调度器
        stages: 包含每个阶段信息的字典列表，每个字典需要有 'epoch' 和 'difficulty' 键
        """
        self.stages = sorted(stages, key=lambda x: x['epoch'])  # 按照epoch排序
        self.current_stage = 0
        self.current_difficulty = self.stages[0]['difficulty'] if self.stages else 'easy'
        self.current_grid_size = self.stages[0]['grid_size'] if self.stages else 4

    def step(self, epoch):
        """
        根据当前epoch更新难度
        """
        # 查找当前应该处于哪个难度阶段
        for i, stage in enumerate(self.stages):
            if epoch >= stage['epoch']:
                self.current_stage = i
            else:
                break
        
        # 更新当前难度和网格大小
        if self.stages and self.current_stage < len(self.stages):
            self.current_difficulty = self.stages[self.current_stage]['difficulty']
            self.current_grid_size = self.stages[self.current_stage]['grid_size']
        
        return self.current_difficulty, self.current_grid_size