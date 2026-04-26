# Records of Observation

## Day 2: OSM Building Retrieval and Visualization
All tests passed.
The shape of the 2D map is pretty accurate, but the height data is not thorough as expected. In "temp_file/47_6059_-122_3392_dlr_group.png", the upper right right sides doesn't have height data, but in Google Map, they are all 3D buildings, including some high-rise. 54% data precision is not good. But it can be a more powerful statement that imperfect data can also create good scene with the power of world model and Mapillary iamges.

### Day 3 Preview:
Day 3 目标（一句话）

从 Mapillary 拉取 Seattle 某坐标周边的街景照片，和 Day 2 的建筑几何显示在同一张图上——你第一次看到"几何 + 视觉"两层数据叠合。

Day 3 会新增的三个概念
1. API token / 环境变量 / .env 文件
Mapillary 需要 API token（免费注册就有）。你会第一次实操"怎么安全地在 code 里用 secret"——不能 commit 到 git，不能硬编码，要用 .env + python-dotenv。这是任何做 API 调用的研究项目都要会的。
你现在可以先做一件事：去申请 Mapillary API token（不用等 Day 3）——https://www.mapillary.com/dashboard/developers，创建一个 app，拿到 access token。从注册到拿到 token 可能有 30 分钟到几小时的审核等待。
2. RESTful API + JSON response
Mapillary API v4 是标准 REST：你发 GET 请求，带 bbox 参数，它回 JSON，里面是一堆图片元数据。你会用 requests 库，处理 pagination（一次拿 100 张，想要更多就翻页），处理速率限制。
Day 2 的 OSM 是 osmnx 帮你封装好了；Day 3 你会第一次直接和 HTTP API 打交道。这是一种更底层的技能。
3. "图像元数据"vs"图像本身"
关键区分：Mapillary API 默认返回的是图片的元数据（ID、GPS、拍摄时间、朝向、缩略图 URL），不是图片本身。如果你想要实际像素，需要再发一次请求去 thumb_2048_url。
Phase 0 Day 3 只拉元数据和缩略图（256px 的小图），不拉大图——原因：

元数据够做可视化（显示拍摄点、朝向）
缩略图够肉眼 sanity check
大图一张 2MB，100 张就 200MB，Week 4 真正要做对齐时再拉

Day 3 的四个步骤
Fetch       从 Mapillary API 拉取坐标附近的图像元数据
  ↓
Filter      去掉方位缺失、距离过远的
  ↓
Reproject   GPS 转成 ENU（复用 Day 1 的 LocalFrame）
  ↓
Visualize   把摄像点画在 Day 2 的建筑图上,箭头表示朝向
最终产出：一个更新版的 visualize_neighborhood.py，画出来的图既有建筑（Day 2），也有 Mapillary 图像的拍摄点 + 朝向箭头（Day 3）。
你会第一次看到"我的查询点周围 50 米内有 23 张真实街景照片，分布在这几个位置"——这是你后面做对齐的原材料。
Day 3 会有几个设计决策要讨论
类似 Day 2 的那种对话：

图像元数据用什么数据结构装？（dataclass + 同样的 anti-corruption 原则）
距离过滤阈值？（太近容易被查询建筑遮挡，太远信息无效）
朝向缺失怎么办？（部分 Mapillary 图片没 compass_angle）
是否下载缩略图到本地？（yes 但要 cache）

这些都是你选，不是我定。