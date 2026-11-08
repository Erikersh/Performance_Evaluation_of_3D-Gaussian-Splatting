# 多重環境變因下的 3D 高斯濺射場景重建效能評估

**專題研究總結報告**

本報告整合兩學期 V1-V11 之實驗流程、拍攝策略、COLMAP / 3D Gaussian Splatting 訓練、量化評估與最終成果。

## 摘要

近年來，3D Gaussian Splatting（3DGS）因具備高品質新視角合成、快速最佳化與即時渲染能力，逐漸成為影像式三維場景重建的重要方法。然而，3DGS 的品質高度依賴輸入影像、相機視角分布，以及前端 Structure-from-Motion（SfM）所估計的相機姿態與初始稀疏點雲，因此在實際室內環境中，拍攝位置、影像重疊、光照、反射、鏡頭視野與相機移動軌跡等因素都可能直接影響最終 PLY 場景與 rendered video。

本研究以室內會議室為主要場景，歷經兩個研究階段。第一階段（V1-V7）建立完整的影像採集、COLMAP、Nerfstudio / Splatfacto 訓練與新視角渲染流程，並探索人工光源、反光、過曝、影像模糊、低紋理、固定拍攝位置與影像數量等問題。其中 V6 以影像品質篩選與 Highlight Mask 改善 COLMAP 輸入，取得 PSNR 25.359 dB、SSIM 0.8845 與 LPIPS 0.2351。

第二階段（V8-V11）將研究焦點由「拍攝後修正」轉向「資料取得策略最佳化」。參考大型影像式三維重建 benchmark 的影片資料形式，本研究改採 iPhone 4K 60 fps 連續影片，再以 FFmpeg 以 1 fps 均勻抽幀，並比較 1× 與 0.5× 視野，以及三圈分層環繞與單圈連續垂直掃描兩種軌跡。V10 取得最高 PSNR 31.228 dB；V11 則取得 PSNR 29.265 dB、SSIM 0.9522、LPIPS 0.0960，並在 geometry completeness 與 rendered room-tour 中呈現最佳整體品質，因此被選定為本研究最終最佳版本。

結果顯示，室內 3DGS 重建品質不能只由影像張數判定。有效多視角資訊量、相鄰影像 overlap、廣視野、近距離物件的重複觀測，以及連續且多高度的相機軌跡，比單純增加照片數量更重要。

**關鍵詞：** 3D Gaussian Splatting、COLMAP、Structure-from-Motion、相機軌跡、影像採集、室內場景重建、新視角合成

# 第一章 緒論

## 1.1 研究背景

3DGS 以大量具位置、尺度、旋轉、透明度與視角相關色彩資訊的三維 Gaussian primitives 表示場景，並透過可微分 rasterization 最佳化。相較於傳統 NeRF，3DGS 在保持高品質新視角合成的同時，具有更高的渲染效率。

但 3DGS 並非完全獨立於前端資料。若輸入影像存在模糊、過曝、強反射、低紋理、視角覆蓋不足或 camera pose 不穩定，COLMAP 的 SfM 結果便可能出現缺失或錯誤，進一步影響 Gaussian scene 的初始化與訓練。

## 1.2 研究動機

研究初期以固定站位、多角度拍照建立會議室資料集。場景中存在白牆低紋理、天花板燈具、高亮區域、自然光、桌椅遮蔽與電視螢幕反射等不利因素。因此第一學期主要關注「光影干擾與影像品質」。

隨著 V1-V7 逐步實驗，團隊發現：影像數量較多不一定帶來更佳結果；固定站位容易產生攝影者附近 blind spot；拍攝位置與相鄰視角 overlap 會明顯改變重建結果。學期間進一步測試大型重建資料集後，研究重心轉向「最佳拍攝方式」，並於第二學期建立 V8-V11 的影片式移動拍攝策略。

## 1.3 研究目的

1. 建立由影像採集、COLMAP SfM、3DGS 訓練、PLY 匯出到 rendered room-tour 的完整流程。

2. 探討光照、反射、過曝、模糊與低紋理對前端重建的影響。

3. 比較固定站位拍照與連續移動式影片採集。

4. 分析影像數量、overlap、Field of View 與 camera trajectory 對重建品質的影響。

5. 比較 1× 與 0.5× 鏡頭。

6. 比較三圈高/中/低分層環繞與單圈連續垂直掃描。

7. 以 PSNR、SSIM、LPIPS、geometry completeness 與 rendered video 共同定義最佳版本。

# 第二章 相關技術

## 2.1 3D Gaussian Splatting

3DGS 使用顯式 Gaussian primitives 進行場景表示，並由 SfM 產生的稀疏三維點初始化。每一 Gaussian 可描述位置、尺度、旋轉、opacity 與球諧函數色彩。由於本研究之 Splatfacto 設定關閉 camera optimizer，因此 COLMAP 所估計的 camera pose 對最終結果具有直接影響。

## 2.2 COLMAP 與 SfM

本研究利用 COLMAP 執行特徵擷取、特徵匹配、相機內外參數估計、incremental reconstruction 與 sparse point cloud 建立。輸出的 cameras.bin、images.bin 與 points3D.bin 供後續 Nerfstudio 使用。

## 2.3 評估指標

| 指標 | 主要意義 | 理論上較佳方向 |
| - | - | - |
| PSNR | 像素層級誤差 | **越高越好** |
| SSIM | 結構、亮度與對比相似性 | **越高、越接近 1 越好** |
| LPIPS | 感知特徵距離 | **越低越好** |
| Standard deviation | 不同 eval views 的離散程度 | 一般越低代表較穩定，但不能單獨決定模型品質 |

本研究不以單一數值決定最佳版本，而採 **Quantitative Metrics + Geometry Completeness + Rendered Video Quality** 綜合判定。

# 第三章 系統架構與實驗環境

## 3.1 整體流程

**第一階段 V1-V7：** iPhone JPG →（部分版本 Filter / Mask）→ COLMAP → Sparse Reconstruction → Nerfstudio / Splatfacto → Gaussian PLY → ns-eval → Virtual Camera Render

**第二階段 V8-V11：** iPhone 4K60 Video → FFmpeg 1 fps 抽幀 → JPG → COLMAP Video Frames → Sparse Reconstruction → Nerfstudio / Splatfacto → Gaussian PLY → ns-eval → Room-tour Render

## 3.2 必要軟硬體配置

- 拍攝：iPhone 14 Plus 或 iPhone 16；同一 Version 內使用同一台裝置。

- V1-V7：固定 1×；JPG。

- V8-V9：1×、4K 60 fps video。

- V10-V11：0.5×、4K 60 fps video。

- SfM：COLMAP。

- 3DGS：Google Colab CUDA GPU、Nerfstudio Splatfacto、gsplat 1.4.0。

- 正式訓練：30,000 iterations。

- config 已確認：Splatfacto、seed 42、camera optimizer off、SH degree 3、SSIM lambda 0.2。

- evaluation split：fraction mode，約 90% train / 10% eval。

# 第四章 第一階段：V1-V7 探索與流程建立

## 4.1 V1：首次固定站位拍攝

241 張 iPhone JPG；開燈、百葉開啟讓自然光進入；電視前固定位置。拍攝角度較隨機、焦距固定，主要目的是確認完整 pipeline 的可行性。未保存正式 evaluation。

## 4.2 V2：提高影像密度

389 張；環境與位置近似 V1。縮小相鄰影像移動與旋轉幅度，提高 overlap，並將電視改為較亮、較不反光的畫面。未保存正式 evaluation。

## 4.3 V3：自然光測試

288 張；關燈、百葉開啟。目標是降低天花板強光造成的過曝、反光與模糊。結果：PSNR 22.906、SSIM 0.8249、LPIPS 0.2714。實驗顯示降低人工光並非單純正向，暗部 texture 同時可能減少。

## 4.4 V4.1：改變 observation center 與補足站位 blind spot

303 張；開燈、自然光進入；拍攝位置移到會議室中段牆前。相鄰影像移動幅度縮小，並在主要拍攝完成後離開原站立位置，反向補拍原本站位周邊。結果：PSNR 24.946、SSIM 0.8719、LPIPS 0.1793。V4.1 為第一階段 LPIPS 最佳版本。

## 4.5 V4.2：更多照片不代表更佳品質

366 張；關燈、自然光；位置與 V3 接近且採更密集拍攝。結果：PSNR 21.576、SSIM 0.8277、LPIPS 0.2950。雖然影像數比 V4.1 多約 21%，三項主要指標卻全面較差，成為「image count ≠ reconstruction quality」的重要證據。

## 4.6 V5：再次補足固定站位附近視角

263 張；開燈、自然光；電視前固定位置。額外向外移動後反向拍攝原站位附近物件。結果：PSNR 23.390、SSIM 0.8790、LPIPS 0.2865。

## 4.7 V6：Filter + Highlight Mask

V6 直接以 V4.1 的 303 張影像為基礎。light_balance.py 將影像分類後保留 276 張，丟棄 27 張。276 張通過影像全部產生對應 Mask；無高光的影像其 Mask 為全白。

主要前處理：

- Laplacian variance：blur score < 15 視為過度模糊。

- Canny edge density：edge density < 0.004 視為低紋理。

- Highlight：高亮像素閾值約 245，若過曝比例達門檻則建立遮罩並以 morphology 擴張忽略區域。

COLMAP 使用 276 張影像與 Mask；3DGS 則使用同一批 276 張「未遮罩原圖」與 sparse model。

結果：PSNR 25.359、SSIM 0.8845、LPIPS 0.2351。PSNR 與 SSIM 為第一階段最佳，但 LPIPS 仍不及 V4.1。這表示 Filter + Mask 對像素與結構一致性有幫助，但不能簡化成「所有視覺品質全面提升」。

## 4.8 V7：低樣本數與中央站位

70 張；開燈、自然光；拍攝位置移至會議室中央。希望平衡各方向距離，但同時大幅降低樣本數。結果：PSNR 13.726、SSIM 0.7105、LPIPS 0.5776，幾何與渲染品質明顯失敗。

V7 能支持的結論是：**在本場景與當時拍法下，70 張影像不足以提供穩定的 SfM / 3DGS 重建。** 由於拍攝位置與樣本數同時改變，不能據此單獨否定中央站位。

# 第五章 第二階段：V8-V11 拍攝策略最佳化

## 5.1 研究方向轉折

學期間測試大型 3D reconstruction datasets 後，團隊發現高品質資料通常具有連續的相機位置變化與均勻取樣特性。因此第二階段不再使用 V6 Filter / Mask，而是從資料取得階段直接改善 camera trajectory 與 FOV。

共同條件：

- 室內燈開啟。

- 窗戶關閉。

- 電視播放內容複雜但盡量低反射的畫面。

- 4K 60 fps video。

- macOS FFmpeg：`ffmpeg -i input.mp4 -vf fps=1 -q:v 2 frame_%04d.jpg`

- COLMAP Data Type 使用 Video Frames。

## 5.2 V8：1× 三圈環繞

iPhone 14 Plus；390 張抽幀 JPG。拍攝者盡量背靠牆面，鏡頭朝空間中央，共繞行三圈，分別取得高、中、低視角。

結果：PSNR 26.612、SSIM 0.9128、LPIPS 0.1752。宏觀房間結構明顯優於固定站位，但靠近拍攝者的桌椅仍出現 blurry / disconnected noise，顯示 1× FOV 在室內近距離環境存在 coverage 盲點。

## 5.3 V9：不同拍攝者與裝置重複實驗

iPhone 16；328 張。拍法盡量維持 V8。結果：PSNR 21.983、SSIM 0.8514、LPIPS 0.2756。其視覺失敗型態與 V8 相近，但量化指標並不相同，因此 V8/V9 不能用來證明手機型號無影響；拍攝者、裝置與最終 frame 數都是混合變因。

## 5.4 V10：0.5× 三圈高/中/低

iPhone 14 Plus；445 張。維持三圈高、中、低結構，但鏡頭改為 0.5×，手機盡量靠近身體以最大化每個 frame 的場景涵蓋。

結果：PSNR **31.228**、SSIM **0.9529**、LPIPS **0.09185**。V10 為全體版本中 PSNR 最高，也在 SSIM、LPIPS 取得最佳或近最佳表現。

## 5.5 V11：0.5× 單圈連續垂直掃描 - 最終最佳版本

iPhone 16；477 張。只沿牆面環繞一圈，但拍攝過程中持續進行高→低→高→低的垂直掃描，一圈約 16 個上下週期，形成「環狀水平移動 + 近似正弦波式垂直視角變化」。

![V10 與 V11 Camera Trajectory 示意圖](visualize\:fde1_bGliZmlsZV9JWmtTOHRvcXJVcjhjYmx4NUFuMXZB_FileDrive_4479a4cb93288191a6300f70c4f9c574)

結果：PSNR 29.265、SSIM 0.9522、LPIPS 0.09596。

雖然 PSNR 略低於 V10，但 SSIM 僅差約 0.00078、LPIPS 僅差約 0.0041；實際 PLY 幾何完整度、近距離物件穩定性與 rendered room-tour 連續性則為團隊觀察中最佳。因此 V11 被選為 **Overall Best Version**。

# 第六章 定量結果與跨版本比較

## 6.1 全部可評估版本

| Version | Images | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| - | -: | -: | -: | -: |
| V3 | 288 | 22.906 | 0.8249 | 0.2714 |
| V4.1 | 303 | 24.946 | 0.8719 | 0.1793 |
| V4.2 | 366 | 21.576 | 0.8277 | 0.2950 |
| V5 | 263 | 23.390 | 0.8790 | 0.2865 |
| V6 | 276 | 25.359 | 0.8845 | 0.2351 |
| V7 | 70 | 13.726 | 0.7105 | 0.5776 |
| V8 | 390 | 26.612 | 0.9128 | 0.1752 |
| V9 | 328 | 21.983 | 0.8514 | 0.2756 |
| V10 | 445 | **31.228** | **0.9529** | **0.09185** |
| V11 | 477 | 29.265 | 0.9522 | 0.09596 |

![V3-V11 量化評估比較圖](visualize\:fde1_bGliZmlsZV9uUE81cnVDQ0dCV2w5OUVDMnhEWnZ3_FileDrive_4479a4cb93288191a6300f70c4f9c574)

V1、V2 未保存正式 evaluation，因此不應補值。

## 6.2 第一階段 vs 第二階段

第一階段較佳版本 V6：PSNR 25.359 / SSIM 0.8845 / LPIPS 0.2351。

V10：PSNR 31.228 / SSIM 0.9529 / LPIPS 0.09185。

V11：PSNR 29.265 / SSIM 0.9522 / LPIPS 0.09596。

由 V6 到 V10，PSNR 提升約 5.87 dB，SSIM 提升約 0.0684，LPIPS 下降約 61%。整體顯示「改善資料取得方式」所帶來的提升大於單純在既有資料上做前端修正。

## 6.3 為何不能只看單一指標

- V10 的 PSNR 最高。

- V11 的整體 geometry / rendered video 最佳。

- V4.1 的 LPIPS 在第一階段優於 V6。

- V8 的 LPIPS 已相當好，但仍存在近距離幾何破碎。

因此 3DGS 場景評估應同時觀察 image-space metrics 與 3D / temporal visual quality。

# 第七章 討論

## 7.1 有效多視角資訊量比照片總數更重要

V4.2 的 366 張影像多於 V4.1 的 303 張，但表現更差。另一方面，V7 僅 70 張時又明顯不足。這說明影像數量存在「最低 coverage 需求」，但超過門檻後，真正重要的是相機位置分布、共同可見區域與特徵可匹配性，而不是單純增加 frame。

## 7.2 固定站位的根本限制

固定站位主要提供方向變化，translation baseline 有限，且攝影者腳下與附近區域容易形成 blind spot。移動式影片採集則讓 camera center 真正在空間中改變，能為同一物件建立更多幾何約束。

## 7.3 0.5× 廣視野在室內環境的作用

室內物件離鏡頭近，1× 時物件容易快速離開畫面。0.5× 讓單張 frame 涵蓋更大範圍，相鄰 frame 共享區域增加，因此 V10/V11 的近距離幾何穩定性明顯提高。

## 7.4 V10 與 V11 的軌跡差異

V10 是三條不同高度的環狀軌跡；V11 則把高度變化整合進單一環狀空間路徑。兩者量化品質非常接近。V11 的價值不是「影像更少」或「拍攝更短」，而是透過連續垂直掃描，使多高度觀察在整個路徑上交錯分布。

# 第八章 研究限制

1. V1-V7 多為 exploratory engineering iterations，部分版本同時改變多個變因。

2. V8/V9 同時存在拍攝者、手機與 frame 數差異，不能用來單獨判定裝置影響。

3. V8→V10 除 1×→0.5× 外，拍攝速度與 frame 數亦不同，因此不能把所有提升完全歸因於 FOV。

4. 第二階段 COLMAP preset 改為 Video Frames，因此跨階段比較同時包含 SfM preset 差異。

5. Geometry completeness 目前主要靠 point cloud、PLY 與 rendered video 人工觀察，尚缺統一客觀分數。

6. 主要實驗場景仍是單一會議室，泛化能力需在其他空間驗證。

# 第九章 結論與未來展望

## 9.1 結論

本研究歷經 V1-V11，完成由固定位置拍照、光影干擾處理，到影片式連續移動與 camera trajectory 最佳化的完整研究演進。

第一階段建立了完整 COLMAP + 3DGS pipeline，並發現光線、反射、影像品質、拍攝位置與視角密度皆會影響結果。V6 以 Filter + Mask 取得第一階段最高 PSNR/SSIM，但仍未解決固定站位的視角覆蓋限制。

第二階段改採 4K60 video + 1 fps 抽幀，並沿空間邊界連續移動。V8/V9 改善宏觀結構，但 1× 對近距離物件仍存在盲點。V10/V11 改用 0.5× 後大幅提升整體品質。最終 V11 以單圈、約 16 次垂直週期掃描的方式，取得非常高的 SSIM / LPIPS，並在實際 PLY 與 room-tour 中呈現最佳整體品質。

**本研究最重要的結論是：高品質室內 3DGS 的關鍵並非單純增加影像張數，而是建立良好的 scene coverage、image overlap、viewpoint diversity、wide FOV 與 continuous camera trajectory。**

## 9.2 建議的最終拍攝流程

1. 使用 0.5× 超廣角。

2. 使用 4K 60 fps 連續影片。

3. 沿空間邊界移動，主要朝向空間中央。

4. 移動時持續改變高低視角，避免只拍單一高度。

5. 保持動作平穩，減少 motion blur。

6. 以 FFmpeg 固定 1 fps 抽幀。

7. COLMAP 使用 Video Frames。

8. 使用 sparse model 初始化 Splatfacto。

9. 以 PSNR / SSIM / LPIPS + Geometry + Rendered Video 綜合評估。

## 9.3 未來工作

- 以同一手機、同一拍攝者做 0.5× vs 1× 單變因比較。

- 比較 0.5 fps、1 fps、2 fps 等抽幀密度。

- 控制步行速度與垂直 sweep frequency。

- 建立 camera coverage heatmap、baseline distribution 與 orientation distribution。

- 加入 registered image ratio、sparse point count、reprojection error 等 SfM 指標。

- 在走廊、教室、狹窄空間與高反射空間進行跨場景驗證。

- 將 V11 軌跡進一步標準化，發展可操作的手機拍攝指引。

# 參考文獻

[1] Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 2023.

[2] Schönberger, J. L., & Frahm, J.-M. Structure-from-Motion Revisited. CVPR, 2016.

[3] Knapitsch, A., Park, J., Zhou, Q.-Y., & Koltun, V. Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction. ACM Transactions on Graphics, 2017.

[4] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on Image Processing, 2004.

[5] Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. CVPR, 2018.

[6] Nerfstudio Project. Splatfacto / data processing and evaluation pipeline documentation.
# Project
