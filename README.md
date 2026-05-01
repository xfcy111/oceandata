# oceandata

高中地理双模式洋流地球页面与公开数据预处理管线。

- 页面入口：https://xfcy111.github.io/oceandata/
- 最新情境 JSON：https://xfcy111.github.io/oceandata/data/yangliu-live-v1.json
- 冬夏固定日期 JSON：https://xfcy111.github.io/oceandata/data/yangliu-seasonal-v1.json

`index.html` 的教学模式使用内联教材洋流路径。情境模式读取预处理 JSON，在浏览器中渲染 SST 覆盖层和 Canvas 流场粒子；默认显示洋流粒子，打开“大气”后主粒子动画切换为 10m 风场。

## Data Workflow

实时数据 workflow：`.github/workflows/build-yangliu-live.yml`

- 洋流：NOAA RTOFS f000 全球海洋场，优先使用公开实时源。
- SST：首版使用 RTOFS 表层温度作为底层。
- 风场：NOAA GFS f000 10m wind。
- 输出：`data/yangliu-live-v1.json`。

季节固定日期 workflow：`.github/workflows/build-yangliu-seasonal.yml`

- 冬季：`2025-01-01`
- 夏季：`2025-07-30`
- 洋流：NOAA RTOFS public S3 2D diagnostic current。
- SST：NOAA OISST daily SST via NCEI ERDDAP。
- 风场：NOAA GFS public S3 f000 10m wind；如果本地缺少 GRIB2/ecCodes 读取能力，脚本会退回 NOAA PSL NCEP/NCAR Reanalysis 10m wind。
- 输出：`data/yangliu-seasonal-v1.json`。

本地生成季节数据：

```bash
python tools/build_yangliu_seasonal_data.py --output data/yangliu-seasonal-v1.json --cache .cache/yangliu-seasonal
```

如果某个公开源失败，JSON 会保留成功图层并在 `errors` 中记录失败原因；前端不会生成假数据，也不会自动跳回教学模式。
