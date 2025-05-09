
**解答1**
a) **平均 $\bar{x}$**
   $$\bar{x} = \frac{1}{5} (1+2+3+4+5) = \frac{15}{5} = 3$$
b) **偏差ベクトル $\mathbf{d}$**
   $$\mathbf{d} = \mathbf{x} - \bar{x}\mathbf{1}_5 = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix} - 3 \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 1-3 \\ 2-3 \\ 3-3 \\ 4-3 \\ 5-3 \end{pmatrix} = \begin{pmatrix} -2 \\ -1 \\ 0 \\ 1 \\ 2 \end{pmatrix}$$
c) **分散 $\sigma^2$**
   偏差の二乗和 $\mathbf{d}^T\mathbf{d} = (-2)^2 + (-1)^2 + 0^2 + 1^2 + 2^2 = 4+1+0+1+4 = 10$
   $$\sigma^2 = \frac{1}{5} (\mathbf{d}^T\mathbf{d}) = \frac{10}{5} = 2$$
   (別公式でも確認: $\mathbf{x}^T\mathbf{x} = 1^2+2^2+3^2+4^2+5^2 = 1+4+9+16+25 = 55$
   $\sigma^2 = (\frac{1}{5} \mathbf{x}^T\mathbf{x}) - \bar{x}^2 = \frac{55}{5} - 3^2 = 11 - 9 = 2$)
d) **標準偏差 $\sigma$**
   $$\sigma = \sqrt{\sigma^2} = \sqrt{2} \approx 1.414$$

---


**解答2**
a) **平均 $\bar{y}$**
   $$\mathbf{1}_4^T \mathbf{y} = \begin{pmatrix} 1 & 1 & 1 & 1 \end{pmatrix} \begin{pmatrix} 10 \\ 20 \\ 30 \\ 40 \end{pmatrix} = 10+20+30+40 = 100$$
   $$\bar{y} = \frac{1}{4} (100) = 25$$
b) **偏差ベクトル $\mathbf{d_y}$**
   $$\mathbf{d_y} = \begin{pmatrix} 10 \\ 20 \\ 30 \\ 40 \end{pmatrix} - 25 \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 10-25 \\ 20-25 \\ 30-25 \\ 40-25 \end{pmatrix} = \begin{pmatrix} -15 \\ -5 \\ 5 \\ 15 \end{pmatrix}$$
c) **分散 $\sigma_y^2$**
   $$\mathbf{d_y}^T \mathbf{d_y} = (-15)^2 + (-5)^2 + 5^2 + 15^2 = 225 + 25 + 25 + 225 = 500$$
   $$\sigma_y^2 = \frac{1}{4} (500) = 125$$

---

**解答3**
データは $x_1=2, x_2=4, x_3=9$。データの個数 $n=3$。
まず、平均 $\bar{x}$ を計算する。
$$\bar{x} = \frac{2+4+9}{3} = \frac{15}{3} = 5$$
次に、データの二乗の和 $\sum x_i^2$ を計算する。
$$\sum x_i^2 = 2^2 + 4^2 + 9^2 = 4 + 16 + 81 = 101$$
「データの二乗の平均」は $\frac{1}{n}\sum x_i^2 = \frac{101}{3}$。
「平均の二乗」は $\bar{x}^2 = 5^2 = 25$。
よって、分散 $\sigma^2$ は、
$$\sigma^2 = \left(\frac{1}{n} \sum x_i^2\right) - \bar{x}^2 = \frac{101}{3} - 25 = \frac{101}{3} - \frac{75}{3} = \frac{26}{3}$$
分散は $\frac{26}{3}$ (約 8.67) です。

---

**解答4**
a) **平均得点 $\bar{s}$**
   $$\bar{s} = \frac{6+8+10}{3} = \frac{24}{3} = 8 \text{ 点}$$
b) **偏差ベクトル $\mathbf{d_s}$**
   $$\mathbf{d_s} = \begin{pmatrix} 6 \\ 8 \\ 10 \end{pmatrix} - 8 \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 6-8 \\ 8-8 \\ 10-8 \end{pmatrix} = \begin{pmatrix} -2 \\ 0 \\ 2 \end{pmatrix}$$
c) **分散 $\sigma_s^2$ と標準偏差 $\sigma_s$**
   偏差の二乗和 $\mathbf{d_s}^T\mathbf{d_s} = (-2)^2 + 0^2 + 2^2 = 4+0+4 = 8$
   $$\sigma_s^2 = \frac{1}{3} (8) = \frac{8}{3} \text{ (点²)}$$
   $$\sigma_s = \sqrt{\frac{8}{3}} = \frac{\sqrt{8}}{\sqrt{3}} = \frac{2\sqrt{2}}{\sqrt{3}} = \frac{2\sqrt{6}}{3} \text{ (点)} \quad (\approx 1.633 \text{ 点})$$
d) **標準化（$z$スコア化）**
   $z_1 = \frac{s_1 - \bar{s}}{\sigma_s} = \frac{6-8}{2\sqrt{6}/3} = \frac{-2}{2\sqrt{6}/3} = \frac{-3}{\sqrt{6}} = \frac{-3\sqrt{6}}{6} = -\frac{\sqrt{6}}{2} \quad (\approx -1.225)$
   $z_2 = \frac{s_2 - \bar{s}}{\sigma_s} = \frac{8-8}{2\sqrt{6}/3} = 0$
   $z_3 = \frac{s_3 - \bar{s}}{\sigma_s} = \frac{10-8}{2\sqrt{6}/3} = \frac{2}{2\sqrt{6}/3} = \frac{3}{\sqrt{6}} = \frac{3\sqrt{6}}{6} = \frac{\sqrt{6}}{2} \quad (\approx 1.225)$
   標準化されたデータベクトル $\mathbf{z} = \begin{pmatrix} -\sqrt{6}/2 \\ 0 \\ \sqrt{6}/2 \end{pmatrix}$
e) **標準化されたデータの平均 $\bar{z}$ と分散 $\sigma_z^2$**
   平均 $\bar{z}$:
   $$\bar{z} = \frac{(-\sqrt{6}/2) + 0 + (\sqrt{6}/2)}{3} = \frac{0}{3} = 0$$
   分散 $\sigma_z^2$:
   偏差ベクトルは $\mathbf{z} - \bar{z}\mathbf{1}_3 = \mathbf{z} - \mathbf{0} = \mathbf{z}$ なので、
   $$\sigma_z^2 = \frac{1}{3} \left( \left(-\frac{\sqrt{6}}{2}\right)^2 + 0^2 + \left(\frac{\sqrt{6}}{2}\right)^2 \right) = \frac{1}{3} \left( \frac{6}{4} + 0 + \frac{6}{4} \right) = \frac{1}{3} \left( \frac{12}{4} \right) = \frac{1}{3} (3) = 1$$
   標準化されたデータの平均は0、分散は1になることが確認できました。

---

**解答5**
a) **平均 $\bar{u}$ と 分散 $\sigma_u^2$**
   $$\bar{u} = \frac{0+0+10+10}{4} = \frac{20}{4} = 5$$
   偏差ベクトル $\mathbf{d_u} = \begin{pmatrix} 0-5 \\ 0-5 \\ 10-5 \\ 10-5 \end{pmatrix} = \begin{pmatrix} -5 \\ -5 \\ 5 \\ 5 \end{pmatrix}$
   偏差の二乗和 $\mathbf{d_u}^T\mathbf{d_u} = (-5)^2 + (-5)^2 + 5^2 + 5^2 = 25+25+25+25 = 100$
   $$\sigma_u^2 = \frac{100}{4} = 25$$
b) **定数 $c=5$ を加えたデータ $\mathbf{v}$ の平均 $\bar{v}$ と分散 $\sigma_v^2$**
   $\mathbf{v} = \mathbf{u} + 5\mathbf{1}_4 = \begin{pmatrix} 0+5 \\ 0+5 \\ 10+5 \\ 10+5 \end{pmatrix} = \begin{pmatrix} 5 \\ 5 \\ 15 \\ 15 \end{pmatrix}$
   $$\bar{v} = \frac{5+5+15+15}{4} = \frac{40}{4} = 10$$
   比較: $\bar{v} = 10$、$\bar{u} = 5$。$\bar{v} = \bar{u} + 5$ となっており、ヒントの $\mu+c$ の関係が成り立っています。
   偏差ベクトル $\mathbf{d_v} = \begin{pmatrix} 5-10 \\ 5-10 \\ 15-10 \\ 15-10 \end{pmatrix} = \begin{pmatrix} -5 \\ -5 \\ 5 \\ 5 \end{pmatrix}$
   （これは元の偏差ベクトル $\mathbf{d_u}$ と同じです）
   偏差の二乗和 $\mathbf{d_v}^T\mathbf{d_v} = (-5)^2 + (-5)^2 + 5^2 + 5^2 = 100$
   $$\sigma_v^2 = \frac{100}{4} = 25$$
   比較: $\sigma_v^2 = 25$、$\sigma_u^2 = 25$。$\sigma_v^2 = \sigma_u^2$ となっており、ヒントの分散は変わらないという関係が成り立っています。
c) **定数 $k=2$ 倍したデータ $\mathbf{w}$ の平均 $\bar{w}$ と分散 $\sigma_w^2$**
   $\mathbf{w} = 2\mathbf{u} = \begin{pmatrix} 2 \times 0 \\ 2 \times 0 \\ 2 \times 10 \\ 2 \times 10 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 20 \\ 20 \end{pmatrix}$
   $$\bar{w} = \frac{0+0+20+20}{4} = \frac{40}{4} = 10$$
   比較: $\bar{w} = 10$、$\bar{u} = 5$。$\bar{w} = 2\bar{u}$ となっており、ヒントの $k\mu$ の関係が成り立っています。
   偏差ベクトル $\mathbf{d_w} = \begin{pmatrix} 0-10 \\ 0-10 \\ 20-10 \\ 20-10 \end{pmatrix} = \begin{pmatrix} -10 \\ -10 \\ 10 \\ 10 \end{pmatrix}$
   （これは元の偏差ベクトル $\mathbf{d_u}$ の各要素を $k=2$ 倍したものです: $k\mathbf{d_u}$）
   偏差の二乗和 $\mathbf{d_w}^T\mathbf{d_w} = (-10)^2 + (-10)^2 + 10^2 + 10^2 = 100+100+100+100 = 400$
   $$\sigma_w^2 = \frac{400}{4} = 100$$
   比較: $\sigma_w^2 = 100$、$\sigma_u^2 = 25$。$\sigma_w^2 = 4 \times \sigma_u^2 = 2^2 \sigma_u^2$ となっており、ヒントの $k^2\sigma^2$ の関係が成り立っています。