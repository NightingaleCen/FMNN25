# Project 3：原版解答与课件的差异

## 检查范围

- 课件：`lecture_opt.pdf`，重点参考第 10–11、15、17、20–21、23、25–34 页。
- 作业要求：[project03.pdf](project03.pdf)。
- 原版代码：提交 `a10e3ee`；同学补充 Task 6 后的版本：提交 `a34102c`。
- 当前整理版：[main.ipynb](main.ipynb) 和 [optimization.py](optimization.py)，已完成 Task 1–9。

这里的“不按课件”指核心算法或约束与课件不同，不把变量命名、类的组织方式、画图样式或示例初始值不同直接算作算法错误。

## 原版中需要调整的题目

同学补充 Task 6 后，明确存在两项独立的方法差异：**Task 4 和 Task 6**。Task 5 使用 Task 4 的线搜索，因此受到连带影响，不重复算作第三项独立问题。

| Task | 原版情况 | 当前整理版 |
|---|---|---|
| 1–2 | 课件未规定具体类结构；未发现方法冲突 | 保留问题类、通用迭代和统一线搜索接口 |
| 3 | Newton 方向、差分 Hessian 和对称化符合要求 | 保留 |
| 4 | 标量搜索未限制步长范围，可能返回负步长 | 使用非负端点构造区间，再做数值最小化 |
| 5 | Rosenbrock 实验本身符合要求，但调用原版 Task 4 | 保留实验，使用修正后的线搜索 |
| 6 | `a10e3ee` 只有说明，尚无实现；`a34102c` 实现的条件和流程不同于课件第 21 页 | 使用课件的 Powell–Wolfe 算法 |
| 7–9 | 上述两个原版提交尚未完成 | 已完成实验和五种 Quasi-Newton 方法 |

### Task 4：exact line search 缺少非负步长约束

课件第 15 页要求沿非负方向搜索：

$$
\alpha_k\in\operatorname*{arg\,min}_{\alpha\geq 0} f(x_k+\alpha s_k).
$$

原版直接调用：

```python
minimize_scalar(lambda alpha: problem(x + alpha * s)).x
```

这次调用没有规定 `alpha >= 0`，所以不保证符合课件的步长约束。即使当前 Rosenbrock 实验得到正步长，也不能证明实现一直满足该约束。

检查用的反例为

$$
f(t)=t^4+5t^3-4t^2-t,\qquad x=0,\quad s=1.
$$

这里 $f'(0)s=-1<0$，是下降方向。原版搜索返回约 `-4.210863`；当前非负方向搜索返回约 `0.565795`。

当前实现先构造非负搜索区间，再使用 Brent 方法数值求解。数值搜索的精度和结果仍取决于函数与区间；不能把它解释为对任意非凸函数的全局最小值保证。

### Task 6：更新版使用另一套 Wolfe 条件和搜索流程

课件第 20–21 页使用 Armijo 条件和 weak Wolfe 曲率条件：

$$
\phi(\alpha)\leq\phi(0)+\sigma\alpha\phi'(0),
\qquad
\phi'(\alpha)\geq\rho\phi'(0).
$$

课件算法的步骤是：

1. 将下端点减半，直到满足 Armijo 条件。
2. 将上端点倍增，直到不满足 Armijo 条件。
3. 用二分调整区间，直到下端点也满足 weak Wolfe 条件。

同学在 `a34102c` 中加入的实现使用 strong Wolfe：

$$
\phi(\alpha)\leq\phi(0)+\rho\alpha\phi'(0),
\qquad
|\phi'(\alpha)|\leq-\sigma\phi'(0).
$$

`rho` 和 `sigma` 的命名与课件交换，本身没有问题。实质差异是使用了绝对值曲率条件，以及另一套 bracketing/sectioning 流程。这种线搜索并非因为不同于课件就数学上错误，但不能称为课件第 21 页算法的直接实现。尚未取得 Fletcher 对应页面，因此没有确认它是否逐步符合书中算法 (2.6.2) 和 (2.6.4)。

更新版另有两处需要说明：

- Markdown 写充分下降参数为 `0.01`，实际测试设为 `1e-4`。课件示例参数为 `sigma=0.01`、`rho=0.9`。
- `alpha_next = 2 * alpha - alpha_prev` 从 `alpha0=1` 出发，会产生 `1, 2, 3, ...` 的等差序列。对 $\phi(\alpha)=(\alpha-10000)^2$ 的检查中，默认 100 次迭代报 `Bracketing failed to converge`，而课件的倍增流程能找到合格步长。

更新版的 Rosenbrock 单独测试可以通过：步长为 `0.0009765625`，Armijo 和 strong Wolfe 条件均成立。通过这一例子不代表两种算法相同。

## 当前完成部分与课件的对应关系

- Task 3：按 Newton 方向求解线性系统，使用差分 Hessian 和对称化；迭代以梯度范数停止，对应第 10–11 页。
- Task 4–5：修正步长约束，保留 exact line search 和 Rosenbrock 实验，对应第 15 页。
- Task 6–7：按第 20–21 页实现并单独检查 Powell–Wolfe 条件，使用课件示例参数。
- Task 8：将相同线搜索接口接入 Newton。先记录初始步长 `2` 时产生非下降方向的情况，再用初始步长 `1` 比较收敛。第 21 页允许任意正初始步长，`2` 只是示例，因此 `1` 不属于算法偏离；没有自动替换方向或修改 Hessian。
- Task 9：good Broyden、bad Broyden、SR1、DFP、BFGS 的更新公式对应第 27、28、30、31、32 页。分母过小或曲率不适合时跳过更新并计数，这是额外的数值处理，不是课件原文中的步骤。

可复用类和函数已移到 `optimization.py`。Notebook 保留说明、实验、结果和图；当前保存了 27 个单元格，13 个代码单元格按顺序执行完成，没有错误输出。

## 与作业文字要求的差异

作业 Task 6 指定 Fletcher 的算法，Task 7 指定该书第 37 页参数。当前整理版按已确认的范围，使用课件第 21 页的算法和示例参数替代。

因此，**符合课件的方法**和**严格符合 Fletcher 原始作业要求**是两个判断。当前说明已明确保留这项差异，不能把课件参数当成已核实的 Fletcher 第 37 页参数。
