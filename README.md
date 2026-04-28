# oceandata

高中地理双模式洋流单页与实时公开数据管线。

- 页面入口：https://xfcy111.github.io/oceandata/
- 数据 JSON：https://xfcy111.github.io/oceandata/data/yangliu-live-v1.json

`yangliu02.html` 的教学模式使用内联教材洋流路径；情境模式读取 GitHub Actions 生成的 `data/yangliu-live-v1.json`，再在浏览器中渲染 SST 覆盖层、Canvas 洋流粒子拖尾和风场辅助箭头。

## Data Workflow

GitHub Actions workflow: `.github/workflows/build-yangliu-live.yml`

- 洋流：NOAA RTOFS f000 全球表层流优先。
- SST：首版使用 RTOFS 表层温度作为兜底层。
- 风场：NOAA GFS f000 10m wind。
- 输出：2° 全球规则网格 `yangliu-live-v1.json`。

如果某个源失败，JSON 会保留成功图层并在 `errors` 中记录失败原因；前端不会自动跳回教学模式，也不会生成假数据。
