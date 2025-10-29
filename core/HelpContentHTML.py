HELP_CONTENT = {
    "general": {
        "title": "程序使用指南",
        "html": """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; }
        h1 { color: #2e7d32; 
            border-bottom: 2px solid #4caf50; }
        h2 { color: #388e3c; }
        h3 { color: black; }
        .code-block { 
            background-color: #e8f5e9; 
            border: 1px solid #c8e6c9;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            overflow-x: auto;
        }
        .param-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        .param-table th {
            background-color: #4caf50;
            color: white;
            text-align: left;
            padding: 8px;
        }
        .param-table td {
            border: 1px solid #c8e6c9;
            padding: 8px;
        }
        .param-table tr:nth-child(even) {
            background-color: #f1f8e9;
        }
        .note {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }
        .math-block {
            font-size: large;
            background-color: #f9fbe7;
            border: 1px solid #dcedc8;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            overflow-x: auto;
        }
        .light-gray {
            color: #888888;
        }
    </style>
    <title>stft_help</title>
</head>
<body>
    <h1>成像数据动态分析工具箱</h1>
    <img src=":/LifeCalor.ico" alt="Logo" style="width:250px;">
    <h2>1. 功能概述</h2>
    <ul>
    <li>全面的数据导入功能，支持多种格式的成像数据导入</li>
        <ul>
        <li>andor相机的sif格式文件序列</li>
        <li>一般的高保真tiff格式文件序列</li>
        <li>avi格式的视频文件</li>
        </ul>
    </li>
    <li>丰富的应用场景（具体请见各方法介绍）</li>
        <ul>
        <li>Fs-iSCAT的载流子动力学分析</li>
        <li>光热信号的处理</li>
        <li>EM-iSCAT细胞电生理分析</li>
        </ul>
    </li>
    <li>强大的可视化成像系统（请见成像模块指南）</li>
        <ul>
        <li>多组数据同时呈现</li>
        <li>自由的ROI绘制功能，匹配各种需求</li>
        <li>可随时应用的伪色样式</li>
        </ul>
    </li>
    <li>充足的导出能力</li>
        <ul>
        <li>数据成像应用样式再导出</li>
        <li>原始数据，处理数据，多种格式均能导出</li>
        <li>结果绘图动态显示以及便捷导出</li>
        </ul>
    </li>
    </ul>
    <h2>2. 界面介绍</h2>
    <ul>
        <li>上方菜单栏：编辑、历史数据、使用指南等核心功能</li>
        <li>左侧工具栏：方法选择、数据导入、参数设置、预处理、后处理、结果导出等功能按钮</li>
        <li>中央成像区：数据成像显示区域，支持ROI绘制，画布增删，样式设置等等</li>
        <li>右侧是结果绘制和控制台显示区</li>
    </ul>

    <h2>3. 使用流程</h2>
    <ol>
        <li>选择分析模式（FS和PA在 <i>超快成像动态分析</i> 里）和数据格式</li>
        <li>导入原始数据</li>
        <li>设置基础的时空参数（如果需要）</li>
        <li class="light-gray">预处理数据（去背景、展数据等操作）（如果有）</li>
        <li>选择想要进行的分析方法</li>
        <li>按照各方法指南设置参数</li>
        <li>按步骤运行分析，等待结果生成</li>
            </ul>
        </li>
        <li>结果可视化与导出：
            <ul>
                <li>在成像区添加需要成像的数据（见成像指南）</li>
                <li>支持多种伪色样式应用</li>
                <li>本步骤可以在任意处理环节应用</li>
            </ul>
        </li>
        <li>导出成像图或结果为需要的格式</li>
    </ol>
</body>
</html>"""
        },
    "canvas":{
        "title": "成像系统指南",
        "html": """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; }
        h1 { color: #2e7d32; 
            border-bottom: 2px solid #4caf50; }
        h2 { color: #388e3c; }
        h3 { color: black; }
        .code-block { 
            background-color: #e8f5e9; 
            border: 1px solid #c8e6c9;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            overflow-x: auto;
        }
        .param-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        .param-table th {
            background-color: #4caf50;
            color: white;
            text-align: left;
            padding: 8px;
        }
        .param-table td {
            border: 1px solid #c8e6c9;
            padding: 8px;
        }
        .param-table tr:nth-child(even) {
            background-color: #f1f8e9;
        }
        .note {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }
        .math-block {
            font-size: large;
            background-color: #f9fbe7;
            border: 1px solid #dcedc8;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            overflow-x: auto;
        }
        .tools-intro {
            border-image: 2px round white;
            margin: 5px;
            border: 2px solid #dcedc8;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        .light-gray {
            color: #888888;
        }
    </style>
    <title>stft_help</title>
</head>
<body>
    <h1>数据可视化利器-成像系统介绍</h1>
    <h2>1. 功能概述</h2>
    为所有已经加载的以及处理过的图像数据进行成像。 
    <ul>
        <li>支持多组图像显示，最多4幅图像同时显示</li>
        <li>支持多种伪色样式应用</li>
        <li>支持多种ROI选取的功能，如线段、矩形、椭圆等</li>
        <li>支持视频数据以及非视频数据</li>
    </ul>
    <h2>2. 工具介绍</h2>
    <div class="note">
        <strong>提示：</strong>鼠标悬停在工具上可以查看工具名称；鼠标右键可以设置其参数（如果有的话）。
    </div>
    <h3>画布工具</h3>
    <div class="tools-intro">   
    <img src=":icons/icon_add.svg" alt="icon_add" style="width:20px;">
     添加画布，点击后选择要显示的数据。
    </div>
    <div class="tools-intro"><img src=":icons/icon_del.svg" alt="icon_del" style="width:20px;">
    删除画布，点击后删除最后添加的画布。
    </div> 
    </div>
    <div class="tools-intro"><img src=":icons/icon_cursor.svg" alt="icon_cursor" style="width:20px;">
    光标工具，选择后会释放所有正在选择的工具，恢复为自由点击状态。
    </div>
        <div class="note">
        <strong>注意：</strong>装盛图像数据的对象称为画布，同时最多只能显示4幅画布。
    </div>
    <h3>ROI绘制工具</h3>
    <div class="tools-intro"><img src=":icons/icon_pen.svg" alt="icon_pen" style="width:20px;">
    画笔。像素ROI工具
    <br>参数：
        <ul>
            <li>设置画笔大小（默认为2px）</li>
            <li>设置画笔颜色</li>
        </ul></div>
    <div class="tools-intro"><img src=":icons/icon_line.svg" alt="icon_line" style="width:20px;">
    直线。像素ROI工具，按住<b>shift</b>键，可限制直线角度
    <br>参数：
        <ul>
            <li>设置画笔大小（默认为2px）</li>
            <li>设置画笔颜色</li>
        </ul></div>
    <div class="tools-intro"><img src=":icons/icon_rect.svg" alt="icon_rect" style="width:20px;">
    矩形。像素ROI工具，按住<b>shift</b>键，可绘制正方形
        <br>参数：
        <ul>
            <li>设置边框粗细（默认为2px）</li>
            <li>设置画笔颜色</li>
        </ul></div>
    <div class="tools-intro"><img src=":icons/icon_ellipse.svg" alt="icon_ellipse" style="width:20px;">
    椭圆。像素ROI工具，按住<b>shift</b>键，可绘制圆形        
    <br>参数：
        <ul>
            <li>设置边框粗细（默认为2px）</li>
            <li>设置画笔颜色</li>
        </ul></div>
    <div class="tools-intro"><img src=":icons/icon_eraser.svg" alt="icon_eraser" style="width:20px;">
    橡皮擦。像素ROI工具，仅用于擦除像素ROI       <br>参数：
        <ul>
            <li>设置橡皮擦大小（默认为2px）</li>
        </ul></div>
            <div class="note">
        <strong>注意：</strong>所有像素ROI工具的参数，比如大小和颜色（不包含填充颜色），是互通的。
    </div>    
    <div class="tools-intro"><img src=":icons/icon_fill.svg" alt="icon_fill" style="width:20px;">
    填充。像素ROI工具，，按住<b>shift</b>键，可限制直线角度        <br>参数：
        <ul>
            <li>设置填充颜色</li>
        </ul></div>
    <div class="tools-intro"><img src=":icons/icon_anchor.svg" alt="icon_anchor" style="width:20px;">
    锚点工具。取点工具，点击后锁定锚点，随着时间轴变化，仅显示锚点处值变化。    
    <br>参数：
        <ul>
            <li>设置颜色</li>
        </ul></div>
    <div class="tools-intro"><img src=":icons/icon_v-line.svg" alt="icon_v-line" style="width:20px;">
    向量直线。向量ROI工具，会沿着向量直线取值，并平均垂直于直线两侧的值，目前仅用于<i>载流子扩散的选取</i>。    
    <br>参数：
        <ul>
            <li>设置选区宽度（默认为2px）<b>此设置为该工具关键设置</b></li>
            <li>设置颜色</li>
        </ul></div>
    <div class="tools-intro"><img src=":icons/icon_v-rect.svg" alt="icon_v-rect" style="width:20px;">
    向量矩形。向量ROI工具，点击后锁定锚点，随着时间轴变化，仅显示锚点处值变化，。    
    <br>参数：
        <ul>
            <li>设置颜色</li>
        </ul></div>
    <h3>其他工具</h3>
    <div class="tools-intro"><img src=":icons/icon_color.svg" alt="icon_color" style="width:20px;">
    样式设置，即应用伪彩色的设置，点击后弹出样式设置对话框
    <br>参数：
        <ul>
            <li>选取需要设置的画布</li>
            <li>设置是否应用伪色（取消勾选的话，便是默认的灰度显示</li>
            <li>选择伪色样式</li>
            <li>设置是否自动设置上下界</li>
            <li>设置显示上下界</li>
        </ul></div>
    <div class="tools-intro"><img src=":icons/icon_accept.svg" alt="icon_accept" style="width:20px;">
    ROI确认，目前除却向量矩形，其他ROI均需要确认，点击后显示画布即ROI列表，双击即可确认</div>
    <div class="tools-intro"><img src=":icons/icon_cancel.svg" alt="icon_cancel" style="width:20px;"><i class="light-gray">现已弃用</i></div>
    <div class="tools-intro"><img src=":icons/icon_reset.svg" alt="icon_reset" style="width:20px;">
    重置，即清空画布上所有物体</div>
    <div class="tools-intro"><img src=":icons/icon_export.svg" alt="icon_export" style="width:20px;">
    导出，点击后进入导出页面，可选择需要导出的画布以及格式</div>

</body>
</html>""",
    },
    "stft": {
        "title": "STFT(短时傅里叶变换)分析",
        "html": """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; }
        h1 { color: #2e7d32; 
            border-bottom: 2px solid #4caf50; }
        h2 { color: #388e3c; }
        h3 { color: black; }
        .code-block { 
            background-color: #e8f5e9; 
            border: 1px solid #c8e6c9;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            overflow-x: auto;
        }
        .param-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        .param-table th {
            background-color: #4caf50;
            color: white;
            text-align: left;
            padding: 8px;
        }
        .param-table td {
            border: 1px solid #c8e6c9;
            padding: 8px;
        }
        .param-table tr:nth-child(even) {
            background-color: #f1f8e9;
        }
        .note {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }
        .math-block {
            font-size: large;
            background-color: #f9fbe7;
            border: 1px solid #dcedc8;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            overflow-x: auto;
        }
    </style>
    <title>stft_help</title>
</head>
<body>
    <h1>STFT（短时傅里叶变换）分析</h1>
    
    <h2>1. 功能概述</h2>
    <p>STFT（Short-Time Fourier Transform）是一种时频分析方法，用于分析信号频率随时间的变化特性。本实现针对电化学调制数据逐像素执行STFT分析，分析数据中周期性变化的频率特性。</p>
    
    <h2>2. 算法原理</h2>
    <p>STFT通过在信号上滑动窗口，对每个窗口内的信号段进行傅里叶变换，从而获得信号在时间和频率上的联合分布：</p>
    <div class="math-block">
    X(τ, ω) = ∫<sub>-∞</sub><sup>∞</sup> x(t) w(t-τ) e<sup>-jωt</sup> dt
    </div>
    <p>其中：</p>
    <ul>
        <li>x(t)：输入信号</li>
        <li>w(t)：窗函数</li>
        <li>τ：时间位置</li>
        <li>ω：角频率</li>
    </ul>
    
    <h2>3. 参数说明</h2>
    <table class="param-table">
        <tr>
            <th>参数</th>
            <th>默认值</th>
            <th>说明</th>
        </tr>
        <tr>
            <td>窗函数类型</td>
            <td>hann</td>
            <td>窗函数类型(如'hann'汉宁窗, 'hamming'汉明窗等，详情查看第七部分)</td>
        </tr>
        <tr>
            <td>目标频率</td>
            <td>30.0 Hz</td>
            <td>想要提取的目标频率</td>
        </tr>
        <tr>
            <td>平均范围</td>
            <td>0 Hz</td>
            <td>默认为0，不平均，否则以目标频率为中心，平均范围内包含的stft结果</td>
        </tr>
        <tr>
            <td>采样帧率</td>
            <td>360 (帧/秒)</td>
            <td>你提供的视频or时序数据所采样的帧率（影响拍摄时长）</td>
        </tr>
        <tr>
            <td>窗口大小</td>
            <td>128 (点数)</td>
            <td>进行短时傅里叶加窗的长度</td>
        </tr>
        <tr>
            <td>窗口重叠</td>
            <td>120 (点数)</td>
            <td>相邻窗重复的长度<br>（步长=窗口大小-窗口重叠）</td>
        </tr>
        <tr>
            <td>变换长度</td>
            <td>360 (点数)</td>
            <td>即参与变换的点数，最小取窗口大小，影响频率点数（非频率分辨率）</td>
        </tr>
    </table>
    
    <h2>4. 处理流程</h2>
    <ol>
        <li>导入原始数据（支持avi和tiff序列）</li>
        <li>预处理数据（去背景、展数据等操作）</li>
        <li>进行质量评价</li>
        <li>逐像素执行STFT：
            <ul>
                <li>提取像素时间序列</li>
                <li>应用窗函数</li>
                <li>计算STFT</li>
                <li>提取目标频率幅度</li>
            </ul>
        </li>
        <li>得到目标频率处具有一定时频分辨率的幅值序列</li>
    </ol>
    <div class="note">
        <strong>注意：</strong>使用前需先执行<i>预处理</i>和<i>质量评估</i>两个步骤
    </div>
    <h2>5. 输出结果</h2>
    <p>质量评价的结果为功率谱密度（PSD）</p>
    <p>STFT处理结果为时序幅值图像（可显示）：</p>
    <div class="code-block">
        stft_py_out[time_index, y_coord, x_coord]
    </div>
    <p>其中：</p>
    <ul>
        <li><strong>time_index</strong>：处理后时间</li>
        <li><strong>y_coord</strong>：像素Y坐标</li>
        <li><strong>x_coord</strong>：像素X坐标</li>
    </ul>
    <h2>6. STFT 时频分辨率浅析</h2>
    <h3>窗口大小（窗长）</h3>
    <p>
        <ul> 
            <li>较长的窗口 → 频率分辨率高（能区分更接近的频率成分），但时间分辨率低（无法精确定位快速变化的瞬态信号）。</li>
            <li>较短的窗口 → 时间分辨率高（能捕捉快速变化），但频率分辨率低（频率模糊）。
        </ul>
        这是由<strong>海森堡不确定性原理</strong>决定的固有权衡。<strong>gabor窗</strong>的特殊之处就在于，它满足了不确定性原理的最下限，保证了时域和频域同时最集中，是使时频图分辨率最高的窗函数。</p>
    <h3>采样率（采样帧率）</h3>
    <p>采样率决定了信号的最高可分析频率（Nyquist频率 = 采样率/2），
        但不会改变STFT的时频分辨率。
        <br>例如：若采样率翻倍，Nyquist频率提高，但窗口长度（以样本点计）不变时，
        实际时间窗口的持续时间（秒）会缩短（因为样本点间隔更小），
        从而可能间接影响时间分辨率。</p>
    <h3>变换步长（窗口重叠）</h3>
    <p>变换步长不会影响实际的时间分辨率，但会决定STFT结果的时间分辨能力。<br>
        <ul>
            <li>较小的步长（高重叠） → 时间采样更密集，但计算量更大。</li>
            <li>较大的步长（低重叠） → 时间采样更稀疏，计算量更小。</li></ul>
        </p>
    <h3>变换长度（nfft）</h3>
    <p>变换长度不会改变实际的频率分辨率，但决定了频率分辨能力（频率点数）。
        <ul>
            <li>较大的变换长度 → 频率点数更多，但计算量更大。</li>
            <li>较小的变换长度 → 频率点数更少，计算量更小。</li></ul>
        当变换长度≥窗口长度时，在计算中尾部会采取补零的操作，频谱插值更平滑，但不会增加真实频率信息。<br>不过在实际测试中，会对结果数值产生一定影响。</p>
    <h2>7. 窗函数介绍</h2>
    <table class="param-table">
        <tr>
            <th>窗函数</th>
            <th>名称</th>
            <th>说明</th>
        </tr>
        <tr>
            <td>hann</td>
            <td>汉宁窗</td>
            <td>默认窗，主瓣较宽，快滚降，频谱泄漏适中</td>
        </tr>
        <tr>
            <td>hamming</td>
            <td>汉明窗</td>
            <td>主瓣适中，旁瓣较低，慢滚降，频谱泄漏适中</td>
        </tr>
        <tr>
            <td>gaussian</td>
            <td>gabor窗</td>
            <td>时间频率分辨率达到理论极限（不确定性原理），旁瓣较低，低泄漏，滚降一般</td>
        </tr>

        <tr>
            <td>boxcar</td>
            <td>矩形窗</td>
            <td>所有点权重相等，主瓣最窄，频谱泄漏严重，滚降最慢</td>
        </tr>
        <tr>
            <td>blackman</td>
            <td>Blackman窗</td>
            <td>主瓣宽，旁瓣低，频谱泄漏很小，滚降快</td>
        <tr>
            <td>blackmanharris</td>
            <td>BH窗</td>
            <td>主瓣超宽，旁瓣极低，频谱泄漏很小，滚降超快</td>
        </tr>
        
    </table>
    <p><i>附：程序使用<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html#scipy.signal.stft">
        scipy.signal.stft</a>方法进行运算</i></p>
</body>
</html>"""
        },
    "cwt": {
        "title": "CWT(连续小波变换)分析",
        "html": """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; }
        h1 { color: #2e7d32; 
            border-bottom: 2px solid #4caf50; }
        h2 { color: #388e3c; }
        h3 { color: black; }
        .code-block { 
            background-color: #e8f5e9; 
            border: 1px solid #c8e6c9;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            overflow-x: auto;
        }
        .param-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        .param-table th {
            background-color: #4caf50;
            color: white;
            text-align: left;
            padding: 8px;
        }
        .param-table td {
            border: 1px solid #c8e6c9;
            padding: 8px;
        }
        .param-table tr:nth-child(even) {
            background-color: #f1f8e9;
        }
        .note {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }
        .math-block {
            font-size: large;
            background-color: #f9fbe7;
            border: 1px solid #dcedc8;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            overflow-x: auto;
        }
    </style>
    <title>stft_help</title>
</head>
<body>
    <h1>CWT(连续小波变换)分析</h1>

<h2>1. 功能概述</h2>
<p>CWT(Continuous Wavelet Transform)是一种先进的时频分析方法，通过不同尺度的小波函数分析信号的局部特征。相比STFT，CWT在时频分辨率上具有更好的灵活性，特别适合分析非平稳信号和瞬态现象。</p>

<h2>2. 算法原理</h2>
<p>CWT通过将信号与小波基函数进行卷积变换：</p>
<p class="math-block">C(a, b) = <sup>1</sup>/<sub>√a</sub> ∫<sub>-∞</sub><sup>∞</sup> x(t) ψ*((t-b)/a) dt</p>
<p>其中：</p>
<ul>
    <li><b>a</b>：尺度参数（与频率相关）</li>
    <li><b>b</b>：平移参数（与时间相关）</li>
    <li><b>ψ(t)</b>：小波基函数</li>
    <li><b>ψ*</b>：小波基函数的复共轭</li>
</ul>
<p>尺度a与频率f的关系为：</p>
<p class="math-block">f = <sup>f<sub>c</sub></sup>/<sub>a</sub></p>
<p>其中f<sub>c</sub>是小波的中心频率。</p>

<h2>3. 参数说明</h2>
<table class="param-table">
    <tr>
        <th>参数</th>
        <th>默认值</th>
        <th>说明</th>
    </tr>
    <tr>
        <td>目标频率</td>
        <td>30.0 Hz</td>
        <td>需要分析的目标频率(Hz)</td>
    </tr>
    <tr>
        <td>小波类型</td>
        <td>cmor</td>
        <td>选择小波基函数类型，详解见下</td>
    </tr>
    <tr>
        <td>采样帧率</td>
        <td>300</td>
        <td>视频或时序数据的采样帧率(帧/秒)</td>
    </tr>
    <tr>
        <td>计算尺度</td>
        <td>256/1</td>
        <td>实际计算频率的数量，影响频率分辨能力（质量评价中256，实际运算中默认为1，即只算目标频率）</td>
    </tr>
    <tr>
        <td>处理跨度</td>
        <td>1</td>
        <td>分析频率范围(Hz)，以目标频率为中心（质量评价中用不到），在这个范围内，会取<i>计算尺度</i>个点进行cwt</td>
    </tr>
</table>

<h2>4. 小波基函数</h2>
<p>目前本程序支持以下小波基函数：</p>
<table class="param-table">
    <tr>
        <th>函数</th>
        <th>名称</th>
        <th>说明</th>
    </tr>
    <tr>
        <td>cmor3-3</td>
        <td>复Morlet小波-宽</td>
        <td>最主要用的小波</td>
    </tr>
    <tr>
        <td>cmor1.5-1.0</td>
        <td>复Morlet小波-窄</td>
        <td>高频率分辨，低时间分辨</td>
    </tr>
    <tr>
        <td>cgau8</td>
        <td>8阶复高斯小波</td>
        <td>对突变敏感，低频率分辨，高时间分辨</td>
    </tr>
        <tr>
        <td>mexh</td>
        <td>墨西哥帽小波</td>
        <td>实小波，对突变敏感，低频率分辨</td>
    </tr>
    <tr>
        <td>morl</td>
        <td>实Morlet小波</td>
        <td>实小波，一般不用，无相位信息</td>
    </tr>
</table>
<h3>cmor3-3 (复Morlet小波)</h3>
<p>复Morlet小波由高斯函数调制复指数构成：</p>
<p class="math-block">ψ(t) = <sup>1</sup>/<sub>√πf<sub>b</sub></sub> e<sup>j2πf<sub>c</sub>t</sup> e<sup>-t²/f<sub>b</sub></sup></p>
<p>其中f<sub>b</sub>=3为带宽参数，f<sub>c</sub>=3为中心频率。</p>
<p><b>特点：</b></p>
<ul>
    <li>提供复数结果（幅度和相位信息）</li>
    <li>良好的时频局部化特性</li>
    <li>适用于精确频率分析</li>
</ul>

<h3>cmor1.5-1.0 (窄带复Morlet小波)</h3>
<p>带宽参数f<sub>b</sub>=1.5，中心频率f<sub>c</sub>=1.0。</p>
<p><b>特点：</b></p>
<ul>
    <li>更高的频率分辨率</li>
    <li>较低的时间分辨率</li>
    <li>适合分析准稳态信号</li>
</ul>

<h3>cgau8 (8阶复高斯小波)</h3>
<p>8阶复高斯小波：</p>
<p class="math-block">ψ(t) = <sup>d⁸</sup>/<sub>dt⁸</sub> (e<sup>-t²/2</sup> e<sup>jωt</sup>)</p>
<p><b>特点：</b></p>
<ul>
    <li>高阶导数提供更好的瞬态检测</li>
    <li>对信号突变敏感</li>
    <li>适合检测边缘和瞬态事件</li>
</ul>

<h3>mexh (墨西哥帽小波)</h3>
<p>墨西哥帽小波是高斯函数的二阶导数：</p>
<p class="math-block">ψ(t) = (1 - t²) e<sup>-t²/2</sup></p>
<p><b>特点：</b></p>
<ul>
    <li>实值小波（仅幅度信息）</li>
    <li>对信号突变高度敏感</li>
    <li>常用于边缘检测和特征提取</li>
    <li>计算效率高</li>
</ul>

<h3>morl(实Morlet小波)</h3>
<p>实部Morlet小波：</p>
<p class="math-block">ψ(t) = e<sup>-t²/2</sup> cos(5t)</p>
<p><b>特点：</b></p>
<ul>
    <li>实数结果（仅幅度信息）</li>
    <li>计算效率高</li>
    <li>适合实时处理</li>
</ul>

<h2>5. 处理流程</h2>
<ol>
    <li><b>数据预处理</b>：加载视频或时序数据，进行降噪和归一化</li>
    <li><b>参数设置</b>：根据信号特性选择小波类型和参数</li>
    <li><b>质量评价</b>：计算得到时频图</li>
    <li><b>CWT计算</b>：对每个像素/通道执行连续小波变换</li>
    <li><b>后处理</b>：后续可以进行全细胞/单细胞分析</li>
    <li><b>可视化</b>：可以对cwt结果进行可视化呈现</li>
</ol>
<h2>7. 时频分辨率分析</h2>
<p>和短时傅里叶变换相比，小波变换有着窗口自适应的特点，即<b>高频信号分辨率高（但是频率分辨率差），低频信号频率分辨率高（但是时间分辨率差）</b></p>

<h3>7.1 不确定性原理</h3>
<p>时频分辨率受海森堡不确定性原理约束：</p>
<p class="math-block">ΔT · ΔF ≥ <sup>1</sup>/<sub>4π</sub></p>
<p>其中ΔT为时间分辨率，ΔF为频率分辨率。</p>

<h3>7.2 不同小波的时频特性</h3>
<table class="param-table">
    <tr>
        <th>小波类型</th>
        <th>时间分辨率</th>
        <th>频率分辨率</th>
        <th>适用场景</th>
    </tr>
    <tr>
        <td>cmor3-3</td>
        <td>中等</td>
        <td>中等</td>
        <td>通用分析</td>
    </tr>
    <tr>
        <td>cmor1.5-1.0</td>
        <td>低</td>
        <td>高</td>
        <td>精确频率分析</td>
    </tr>
    <tr>
        <td>cgau8</td>
        <td>很高</td>
        <td>很低</td>
        <td>边缘和突变检测</td>
    </tr>
    <tr>
        <td>mexh(实)</td>
        <td>高</td>
        <td>低</td>
        <td>特征提取/边缘检测</td>
    </tr>
    <tr>
        <td>morl(实)</td>
        <td>高</td>
        <td>低</td>
        <td>瞬态事件检测</td>
    </tr>
</table>

<!-- <h3>7.3 尺度选择的影响</h3>
<ul>
    <li><b>尺度数量多</b>：频率分辨率高，计算时间长</li>
    <li><b>尺度数量少</b>：频率分辨率低，计算时间短</li>
    <li><b>处理跨度大</b>：分析频率范围宽，分辨率降低</li>
    <li><b>处理跨度小</b>：分析频率范围窄，分辨率提高</li>
</ul> -->

    <p><i>附：程序使用<a href="https://pywavelets.readthedocs.io/en/latest/ref/cwt.html">
        pywt.cwt</a>方法进行运算</i></p>
</body>
</html>"""
        },
    "lifetime": {
        "title": "指数型寿命计算（没写完）",
        "html": """
<h2>指数型寿命计算</h2>

<h3>1. 功能概述</h3>
<p>指数型寿命计算基于Arrhenius方程，用于预测材料或设备在特定条件下的使用寿命。该方法广泛应用于电子元件、化工材料和机械部件的寿命评估。</p>

<h3>2. 计算原理</h3>
<p>寿命计算公式：</p>
<p class="math-block">L = A × e<sup>(Ea/(k×T))</sup></p>
<p>其中：</p>
<ul>
    <li>L：预测寿命</li>
    <li>A：前置因子</li>
    <li>Ea：活化能(eV)</li>
    <li>k：玻尔兹曼常数(8.617333262145×10<sup>-5</sup> eV/K)</li>
    <li>T：绝对温度(K)</li>
</ul>

<h3>3. 参数说明</h3>
<table>
    <tr>
        <th>参数</th>
        <th>类型</th>
        <th>说明</th>
    </tr>
    <tr>
        <td>activation_energy</td>
        <td>float</td>
        <td>活化能(eV)</td>
    </tr>
    <tr>
        <td>pre_exponential</td>
        <td>float</td>
        <td>前置因子</td>
    </tr>
    <tr>
        <td>temperature</td>
        <td>float</td>
        <td>工作温度(℃)</td>
    </tr>
    <tr>
        <td>stress_levels</td>
        <td>list</td>
        <td>应力水平列表</td>
    </tr>
</table>

<h3>4. 使用步骤</h3>
<ol>
    <li>收集加速寿命试验数据</li>
    <li>确定活化能Ea</li>
    <li>计算前置因子A</li>
    <li>输入工作温度</li>
    <li>计算预测寿命</li>
</ol>
"""
        },
    "whole": {
        "title":"全细胞分析（没写完）",
        "html": """cwt_help.html<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; }
        h1 { color: #2e7d32; 
            border-bottom: 2px solid #4caf50; }
        h2 { color: #388e3c; }
        h3 { color: black; }
        .code-block { 
            background-color: #e8f5e9; 
            border: 1px solid #c8e6c9;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            overflow-x: auto;
        }
        .param-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        .param-table th {
            background-color: #4caf50;
            color: white;
            text-align: left;
            padding: 8px;
        }
        .param-table td {
            border: 1px solid #c8e6c9;
            padding: 8px;
        }
        .param-table tr:nth-child(even) {
            background-color: #f1f8e9;
        }
        .note {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }
        .math-block {
            font-size: large;
            background-color: #f9fbe7;
            border: 1px solid #dcedc8;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            overflow-x: auto;
        }
    </style>
    <title>stft_help</title>
</head>
<body>
    <h1>全细胞电生理信号提取与分析</h1>

没写完，有问题问我
</body>
</html>""",
        },
    "single": {
        "title":"单通道分析（没写完）",
        "html": """cwt_help.html<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; }
        h1 { color: #2e7d32; 
            border-bottom: 2px solid #4caf50; }
        h2 { color: #388e3c; }
        h3 { color: black; }
        .code-block { 
            background-color: #e8f5e9; 
            border: 1px solid #c8e6c9;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            overflow-x: auto;
        }
        .param-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        .param-table th {
            background-color: #4caf50;
            color: white;
            text-align: left;
            padding: 8px;
        }
        .param-table td {
            border: 1px solid #c8e6c9;
            padding: 8px;
        }
        .param-table tr:nth-child(even) {
            background-color: #f1f8e9;
        }
        .note {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }
        .math-block {
            font-size: large;
            background-color: #f9fbe7;
            border: 1px solid #dcedc8;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            overflow-x: auto;
        }
    </style>
    <title>stft_help</title>
</head>
<body>
    <h1>单通道电生理信号提取与分析</h1>

没写完，有问题问我
</body>
</html>""",
        },
    }

