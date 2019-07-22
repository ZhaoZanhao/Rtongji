# 初识R

# 输入体重的数据
X1 <- c(35, 40, 40, 42, 37, 45, 43, 37, 44, 42, 41, 39)
mean(X1) # 计算体重均值
sd(X1) # 计算体重的标准差
# 输入胸围数据
X2 <- c(60, 74, 64, 71, 72, 68, 78, 66, 70, 65, 73, 75)
mean(X2) # 计算胸围的均值
sd(X2) # 计算胸围的标准差
# 绘制散点图
plot(X1, X2)
# 绘制直方图
hist(X1)
hist(X2)


# 向量
x <- c(10.4, 5.6, 3.1, 6.4, 21.7)
assign("x", c(10.4, 5.6, 3.1, 6.4, 21.7))
x
y <- c(x, 0, x)
y

# 向量运算
x <- c(-1, 0, 2); y <- c(3, 8, 2)
v <- 2*x + y +1; v
 
# 运算
x * y
x / y
x^2
y^x

5%/%3 # 整除
5%%3 # 取余数

exp(x)
sqrt(y)

x <- c(10, 6, 4, 7, 8)
min(x)
max(x)
range(x)
which.max(x)
which.min(x)
sum(x) # 求和
median(x) # 中位数
mean(x) # 平均数
var(x) # 方差
sd(x) # 标准差
length(x) # 向量的长度
sort(x) # 对向量进行排序
order(x) # 对向量进行排序的下标
sort.list(x) # 对向量进行排序的下标

prod(x) # 未知

# 产生有规律的序列
x <- 1:30
x

# 等差序列
# 当a为实数，b为整数时，向量a:b是实数
# 当a为整数，b为实数时，向量a:b是整数


2.312 : 6
4 : 7.6

x <- 2*1:15
x

# 等差运算 优于 乘法运算
n <- 5
1:n-1
1:(n-1)

# 等间隔函数
# seq()函数，产生等距离函数
s1 <- seq(-5, 5, 0.2)
s1
s2 <- seq(length=51, from=-5, by=0.2)
s2

# 重复函数
# rep()是重复函数
x <- c(1, 4, 6.25); x
s <- rep(x, times=3)
s

# 逻辑向量
x <- 1:7
L <- x>3
L

# 逻辑变量的赋值
z <- c(TRUE, FALSE, F, T)
z

# 判断一个逻辑向量是否都为真值的函数
all(c(1, 2, 3, 4, 5, 6, 7)>3)

# 判断一个逻辑向量其中是否有真值的是any
any(c(1, 2, 3, 4, 5, 6, 7)>3)

# 缺失数值
z <- c(1:3, NA); z
ind <- is.na(z)
ind

# 将缺失数据改为0
z[is.na(z)] <- 0; z

# is.nan()检测数据是否不确定，T表示确定
# is.finite()检测数据是否有限，T表示有限
# is.infinite()检测数据是否无穷，T表示无穷

x <- c(0/1, 0/0, 1/0, NA); x
is.nan(x)
is.finite(x)
is.infinite(x)
is.na(x)

# 字符型向量
y <- c("er","sdf","eir","jk","dim")
labs <- paste("X", 1:6, sep=""); labs

paste("result.", 1:4, sep="")

paste(1:10)
paste("Today is", date())
paste(c("a","b"), collapse = '.')

# 复数向量
x <- seq(-pi, pi, pi/10)
x
y <- sin(x)
y
z <- complex(re=x, im=y)
plot(z)
lines(z)

# 向量下标运算
x <- c(1, 4, 7)
x[2]

(c(1,3,5)+5)[2]

x[2] <- 125
x

x[c(1,3)] <- c(144,169)
x

# 逻辑运算
x <- c(1,4,7)
x<5
x[x<5]

# 将向量中的缺失数据赋值为0
z <- c(-1, 1:3, NA)
z

z[is.na(z)]<-0
z

z <- c(-1, 1:3, NA)
y <- z[!is.na(z)]
y

z+1
x <- (z+1)[(!is.na(z)) & z>0]
x


y <- numeric(length(x))
y[x<0] <- 1-x[x<0]
y[x>=0] <- 1+x[x>=0]


# 下标的正整数运算
v <- 10:20
v[c(1,3,5,9)]
v[1:5]
v[c(1,2,3,2,1)]
c("a","b","c")[rep(c(2,1,3),times=3)]

# 下标的负整数运算
v
# v作为一个向量，下标取值在-length(z)到-1之间
v[-(1:5)]

# 取字符型值的下标向量
ages <- c(Li=33, Zhang=29, Liu=18)
ages

ages["Zhang"]

fruit <- c(5, 10, 1, 20)
names(fruit) <- c("orange", "banana", "apple", "peach")
fruit


# 对象和她的模式与属性
# 固有属性
# 类型属性和长度属性
# 类型属性
# logical逻辑型
# numeric数值型
# complex复数型
mode(c(1,3,5)>5)

z <- 0:9
is.numeric(z)
is.character(z)

length(2:4)
length(z)
# 进行强制类型转换
digits <- as.character(z); digits
d <- as.numeric(digits); d

# attributes()和attr()函数
x <- c(apple=2.5, orange=2.1); x
attributes(x)
attr(x,"names")

# attr()函数写作赋值左边以改变属性值或者定义新的属性
attr(x, "names") <- c("apple", "grapes"); x
attr(x, "type")  <- "fruit"; x
attr(x, "type")

attributes(x)

# factor()函数
sex <- c("M", "F", "M", "M", "F")
sexf <- factor(sex);
sexf

sex.level <- levels(sexf)
sex.level

# 对于因子向量，可以用函数table()来统计各类数据的频数
sex.tab <- table(sexf)
sex.tab

# tapply()函数
height <- c(174, 165, 180, 174, 160)

tapply(height, sex, mean)

# 多维数组和矩阵

# 将向量定义成数组
z <- 1:12
dim(z) <- c(3,4)
z
# 矩阵中的元素默认是按照列存放的

# 使用array()函数构造数组
dim(z) <- 12
z

# 使用array()函数构造多维数组
X <- array(1:20, dim = c(4,5))
X

# 常用如下方法对数组进行初始化

Z <- array(0, dim = c(3, 4, 2))
Z

# 用matrix构造矩阵

A <- matrix(1:15, nrow=3, ncol=5, byrow = F)
A

A <- matrix(1:15, nrow=3, ncol=5, byrow=T)
A


# 数组下标
a <- 1:24
a
dim(a) <- c(2,3,4)
a

# 确定下标位置
a[2,1,2]

a[1, 2:3, 2:3]

a[1, , ]

a[, , 2]

a[1, 1, ]

a[]

a[ , , ]


a[3:10]


b <- matrix(c(1,1,1, 2,2,3, 1,3,4, 2,1,4), ncol=3, byrow=T)
b



a[b]

a[b] <- 0

a



# 数组之间的四则运算
A <- matrix(1:6, nrow=2, byrow = T)
A

B <- matrix(1:6, nrow = 2, byrow = F)
B

C <- matrix(c(1,2,2,3,3,4), nrow = 2)
C

D <- 2*C + A/B
D

# 不规则数据的匹配
x1 <- c(100, 200)
x2 <- 1:6
x1+x2
x3 <- matrix(1:6, ncol=3, byrow=T)
x1+x3
x2 <- 1:5
x1+x2


# 矩阵的运算
A <- matrix(1:6, nrow = 2)
A

# 转置运算
t(A)

# 求矩阵的行列式
det(matrix(1:4, ncol = 2))

# 求向量的内积
x <- 1:5
y <- 2*1:5
x%*%y
1^2+2^2+3^2+4^2+5^2+6^2

# 向量内积crossprod()
crossprod(x)
x%*%y

# 向量外积tcrossprod()
tcrossprod(x)
x%o%y
outer(x,y)
# outer()在绘制三维曲面的时候非常有用


# 矩阵乘法
A <- array(1:9, dim=c(3,3))
B <- array(9:1, dim=c(3,3))

C <- A*B
C
A
B

D <- A%*%B
D

# 生成对角矩阵
v <- c(1,4,7)
diag(v)

M <- array(1:9, dim=c(3,3))
diag(M)


# 解线性方程组和求解矩阵的逆矩阵
A <- t(array(c(1:8, 10), dim=c(3,3)))
A

b <- c(1,1,1)
b

# Ax=b的解
x <- solve(A, b)
x

# 求解A的逆矩阵B
B <- solve(A)
B

# 求解矩阵的特征值和特征向量
# 函数eigen()
Sm <- crossprod(A,A)
Sm

ev <- eigen(Sm)
ev

# 矩阵的奇异值分解
# 函数svd()

svdA <- svd(A)
svdA

attach(svdA)
u %*% diag(d) %*% t(v)

# 求解矩阵行列式的值
det(A)

# 最小拟合与QR分解
# lsfit()函数
x <- c(0.0, 0.2, 0.4, 0.6, 0.8)
y <- c(0.9, 1.9, 2.8, 3.3, 4.2)
lsfit.sol <- lsfit(x,y)
lsfit.sol

# $coefficients是拟合的系数
# $residuals是拟合残差
# $intercept

# 与lsfit()函数相密切的函数ls.diag()
X <- matrix(c(rep(1,5),x), ncol = 2)
X
Xplus <- qr(X)
Xplus

# 使用QR分解得到结果的最小二乘的系数
b <- qr.coef(Xplus,y)
b

# 通过QR分解计算得到最小二乘拟合的拟合值和残差值
fit <- qr.fitted(Xplus,y)
fit
res <- qr.resid(Xplus,y)
res

# 与数组有关的函数
A <- matrix(1:6, ncol=3)
A
dim(A)
dim(A)[1]
dim(A)[2]
nrow(A)
ncol(A)

# 矩阵的合并
# cbind()横向合并
# rbind()纵向合并
x1 <- rbind(c(1,2), c(3,4))
x1

x2 <- cbind(x1, x1)
x2

x3 <- rbind(x1, x1)
x3

x4 <- cbind(1, x1)
x4


# 矩阵的拉直
# as.vector()函数
A <- matrix(1:6, ncol=2)
A

as.vector(A)
b <- as.vector(A)
b

# 数组的维的名字
# 数组有个属性dimnames保存各个维度各个下标的名字，只是一个标签而已
X <- matrix(1:6, ncol=2, dimnames=list(c("one","two","three"), c("First","Second")), byrow=T)
X
X[,1]


# 可以先定义矩阵，然后再进行维度标签命名
X <- matrix(1:6, ncol=2, byrow=T)
X
dimnames(X) <- list(c("one", "two", "three"), c("First", "Second"))
X

# 对于矩阵，还可以使用属性rownames()和colnames()来访问
X <- matrix(1:6, ncol=2, byrow=T)
X
rownames(X) <- c("one", "two", "three")
colnames(X) <- c("First", "Second")
X

# 数组的广义转置
A <- array(1:24, dim=c(2,3,4))
A

B <- aperm(A, c(2,3,1))
B

# apply函数
A <- matrix(1:6, nrow=2, byrow=T)
A

apply(A, 2, mean)
apply(A, 1, sum)

# 列表与数据框
lst <- list(name="Fred", wife="Marry", no.children=3, child.ages=c(4,7,9))
lst
# 列表元素总可以用列表名[[下标]]的格式引用
lst[[2]]
lst[[4]][2]

# 子列表
lst[1:2]

lst[["name"]]
lst[["child.ages"]]
lst[["wife"]]

# 列表的修改
lst$name <- "john"
lst$name
lst$income <- c(3000,3500)
lst

# 如果要删除列表里面的某一项
# 就给这一项附一个空值
lst$income <- NULL
lst

lst.abc <- c(lst$name, lst$wife, lst$child.ages)
lst.abc

# 数据框
# 数据框的生成
df <- data.frame(
  name=c("Alice", "Becka", "James", "Jeffrey", "Jhon"),
  sex=c("F", "M", "F", "M", "M"),
  ages=c(12,23,24,15,18),
  height=c(156,158,159,160,159),
  weight=c(58,50,56,52,55)
)

df

# 如果一个列表的所有各个成分都满足数据框的要求
# 可以使用as.data.frame()函数强制转化为数据框
lst <- list(name=c("Alice", "Becka", "James", "Jeffrey", "Jhon"),
            sex=c("F", "M", "F", "M", "M"),
            ages=c(12,23,24,15,18),
            height=c(156,158,159,160,159),
            weight=c(58,50,56,52,55)
            )
lst

df1 <- as.data.frame(lst)
df1

# 一个矩阵可以成为一个数据框
# 原来的列名就是数据框的变量名
# 否则系统将自动为矩阵起一个列名
X <- array(1:6, dim = c(2,3))
X
df2 <- as.data.frame(X)
df2

# 数据框的引用
# 使用下标或者下标向量
df[1:2, 3:5]

# 数据框的各个变量也可以用[[]]或者$进行引用
df[["name"]]
df$ages

# 数据框的变量名由names()定义
names(df)
# 数据框的观测名由rownames()定义
rownames(df) <- c("one", "two", "three", "four", "five")
df

# R提供了attach()函数将数据框的每一个变量“连接”到内存中，便于数据框的调用
df <- data.frame(
  name=c("Alice", "Becka", "James", "Jeffrey", "Jhon"),
  sex=c("F", "M", "F", "M", "M"),
  ages=c(12,23,24,15,18),
  height=c(156,158,159,160,159),
  weight=c(58,50,56,52,55)
)
attach(df)
r <- height/weight
r
df$r <- height/weight
df

# 为了取消连接
# 使用detach()函数
# 没有参数即可
detach()


# 列表与数据框的编辑
# 如果需要对数据框或者列表中的数据进行修改编辑，可以使用edit()函数执行

# read.table()函数
# 读入文本文件
rt <- read.table("houses.data.txt")
rt

is.data.frame(rt)

rt1 <- read.table("houses.data.txt", header=TRUE)
rt1

# scan函数：能够直接读取出纯文本的数据
w <- scan("test.txt")
w

# 假设数据中有不同属性
inp <- scan("h_w.txt", list(height=0, weight=0))
inp
is.list(inp)
inp$weight

X <- matrix( scan("h_w.txt", list(height=0, weight=0))$weight, ncol=5, nrow=3, byrow = T)
X

X1 <- matrix(scan("h_w.txt"), nrow=6, ncol=5, byrow=T)
X1


x <- scan()
1

# 读取其他软件的数据
library(foreign)
library(sas7bdat)
sasdf <- read.sas7bdat("F:\\kpmg\\ifs9\\luojiku\\allk.sas7bdat")
head(sasdf)

# 使用haven包读取SAS明显比sas7bdat包要快
library(haven)
sasdata <- read_sas("F:\\kpmg\\ifs9\\luojiku\\allk.sas7bdat")
sasdata

# 一种导入EXCEL文件的方式
library(readxl)
exceldf <- read_excel("F:\\kpmg\\data\\vintage.xlsx")
exceldf

# 链接嵌入的数据库
data("infert")

# 写数据文件
# write()函数
df <- data.frame(
  Name=c("Alice", "Becka", "James", "Jeffrey", "John"),
  Sex=c("F", "F", "M", "M", "M"),
  Age=c(13, 13, 12, 13, 12),
  Height=c(56.5, 65.3, 57.3, 62.5, 59.0),
  Weight=c(84.0, 98.0, 83.0, 84.0, 99.5)
)

write.csv(df, file="foo.csv")

write.table(df, file="foo.txt")

# 控制流
# if else语句
# switch()语句
y <- "fruit"
switch(y, fruit="banan", vegetable="broccoal", meet="beef")

# 中止语句break()
# 空语句next()

# 循环语句for()语句、while()循环、repeat()语句
# 少写循环
n <- 4
x <- array(0, dim=c(n,n))
x

for (i in 1:n)
{
  for (j in 1:n)
  {
    x[i,j] <- 1/(i+j-1)
  }
}
x

# while循环
f <- 1
f[2] <- 1
f
i <- 1

while(f[i]+f[i+1]<1000)
{
  f[i+2] <- f[i] + f[i+1]
  i <- i+1
}
f

# repeat循环
f <- 1
f[2] <- 1
i <- 1
repeat
{
  f[i+2] <- f[i] + f[i+1]
  i <- i+1
  if (f[i] + f[i+1] >= 1000) break 
}

f

# repeat循环
f <- 1
f[2] <- 1
i <- 1
repeat
{
  f[i+2] <- f[i] + f[i+1]
  i <- i+1
  if (f[i] + f[i+1] < 1000) next else break 
}

f


# 编写自己的函数
# 二分法
fzero <- function(f, a, b, eps)
{
  if (f(a)*f(b)>0)
    list(fail="finding root is failing!")
  else
  {
    repeat
    {
      if (abs(a-b)<eps) break
      x <- (a+b)/2
      if (f(a)*f(x)<0) b<-x else a<-x
    }
    list(root=(a+b)/2, fun=f(x))
  }
}

# 建立自己的函数
f <- function(x) x^3-x-1
fzero(f, 1, 2, 1e-6)

# 求一元函数的根
uniroot(f, c(1,2))

# 自己定义函数
twosam <- function(y1, y2)
{
  n1 <- length(y1)
  n2 <- length(y2)
  yb1 <- mean(y1)
  yb2 <- mean(y2)
  s1 <- var(y1)
  s2 <- var(y2)
  s <- ((n1-1)*s1+(n2-1)*s2)/(n1+n2-2)
  (yb1-yb2)/sqrt(s*(1/n1+1/n2))
}

A <- c(79.98, 80.04, 80.02, 80.04, 80.03, 80.03, 80.04, 79.97, 80.05, 80.03, 80.02, 80.00, 80.02)
B <- c(80.02, 79.94, 79.98, 79.97, 79.97, 80.03, 79.95, 79.97)
twosam(A, B)

# 定义新的二元计算
"%!%" <- function(x, y) {exp(-0.5*(x-y) %*% (x-y))}
x <- 1
y <- 1
x%!%y

# 有名参数与缺省
Newtons <- function(funs, x, ep=1e-5, it_max=100)
{
  index <- 0
  k <- 1
  while(k<=it_max)
  {
    x1 <- x
    obj <- funs
    x <- x-slove(obj$J, obj$f)
    norm <- sqrt((x-x1) %*% (x-x1))
    if (norm<ep)
    {
      index <- 1
      break
    }
    k <- k+1
  }
  obj <- funs(x)
  list(root=x, it=k, index=index, FunVal=obj$f)
}

# 上一个函数的参数
funs<-function(x)
{
  f<-c(x[1]^2+x[2]^2-5, (x[1]+1)*x[2]-(3*x[1]+1))
  J<-matrix(c(2*x[1], 2*x[2], x[2]-3, x[1]+1), nrow=2, byrow=T)
  list(f=f, J=J)
}

# 求解下列方程
Newtons(funs, c(0,1))

# 递归函数
area <- function(f, a, b, eps = 1.0e-06, lim = 10) 
{
  fun1 <- function(f, a, b, fa, fb, a0, eps, lim, fun) 
  {
    d <- (a + b)/2; h <- (b - a)/4; fd <- f(d)
    a1 <- h * (fa + fd); a2 <- h * (fd + fb)
    if(abs(a0 - a1 - a2) < eps || lim == 0)
      return(a1 + a2)
    else {
      return(fun(f, a, d, fa, fd, a1, eps, lim - 1, fun) + fun(f, d, b, fd, fb, a2, eps, lim - 1, fun))
          }
  }
  fa <- f(a); fb <- f(b); a0 <- ((fa + fb) * (b - a))/2
  fun1(f, a, b, fa, fb, a0, eps, lim, fun1)
}

# 先定义函数
f <- function(x) 1/x
quad <- area(f, 1, 5)
quad

# 第三章
# 数据的描述性分析
w <- c(75.0, 64.0, 47.4, 66.9, 62.2, 62.2, 58.7, 63.5, 66.6, 64.0, 57.0, 69.0, 56.9, 50.0, 72.0)
w.mean <- mean(w)
w.mean

x <- 1:12
dim(x) <- c(3,4)
x
mean(x)

# 如果需要矩阵每行或者每列的均值
apply(x, 1, mean)
apply(x, 2, mean)

mean(as.data.frame(x))

# 顺序统计量
x <- c(75, 64, 47.4, 66.9, 62.2, 62.2, 58.7, 63.5)
sort(x)
sort(x, decreasing = TRUE)

x.na <- c(75.0,64.0,47.4,NA,66.9,62.2,62.2,58.7,63.5)
sort(x.na)
x.na
sort(x.na, na.last = T)
sort(x.na, na.last = F)


# 与sort有关的函数有
# order()给出排序后的坐标
order(x)
# rank()函数给出样本的秩
rank(x)

# 中位数
x <- c(75,64,47.4,66.9,62.2,58.7,63.5)
median(x)

# 当na.rm=TRUE时可以处理缺失变量
x.na <- c(75,64,NA,47.4,66.9,62.2,58.7,63.5)
median(x.na, na.rm = FALSE)
median(x.na, na.rm = TRUE)

# 百分位数
w <- c(75,64,47.4,66.9,62.2,62.2,58.7,63.5,66.6,64.0,57,69,56.9,50,72)
quantile(w)
seq(0,1,0.2)
quantile(w, probs = c(seq(0,1,0.2)))

# 分散程度度量
# 数据分散/变异程度的特征量有：方差、标准差、极差、四分位极差、变异系数、标准误差
w <- c(75,64,47.4,66.9,62.2,62.2,58.7,63.5,66.6,64.0,57,69,56.9,50,72)
# 方差
var(w)
sd(w)
cv <- 100*sd(w)/mean(w);cv
css <- sum((w-mean(w))^2);css
uss <- sum(w^2);uss

# 输出各种统计量
data_outline <- function(x)
{
  n <- length(x)
  m <- mean(x)
  v <- var(x)
  s <- sd(x)
  me <- median(x)
  cv <- 100*s/m
  css <- sum((x-m)^2)
  uss <- sum(x^2)
  R <- max(x)-min(x)
  R1 <- quantile(x,3/4)-quantile(x,1/4)
  sm <- s/sqrt(n)
  g1 <- n/((n-1)*(n-2))*sum((x-m)^3)/s^3
  g2 <- ((n*(n+1))/((n-1)*(n-2)*(n-3))*sum((x-m)^4)/s^4 - (3*(n-1)^2)/((n-2)*(n-3)))
  data.frame(N=n, Mean=m, Var=v, std_dev=s, Median=me, std_mean=sm, CV=cv, CSS=css, 
             USS=uss, R=R, R1=R1, Skewness=g1, Kurtosis=g2, row.names=1)
}

w <- c(75.0, 64.0, 47.4, 66.9, 62.2, 62.2, 58.7, 63.5, 66.6, 64.0, 57.0, 69.0, 56.9, 50.0, 72.0)
data_outline(w)


# 数据的分布

# 直方图、经验分布图、QQ图

W <- c(75.0, 64.0, 47.4, 66.9, 62.2, 62.2, 58.7, 63.5, 66.6, 64.0, 57.0, 69.0, 56.9, 50.0, 72.0)
hist(W, freq=F)
lines(density(W), col="blue")
x <- 44:76
lines(x, dnorm(x, mean(W), sd(W)), col = "red")

# 经验分布
# 在R中，用函数ecdf()绘制出样本的经验函数
w <- c(75.0, 64.0, 47.4, 66.9, 62.2, 62.2, 58.7, 63.5, 66.6, 64.0, 57.0, 69.0, 56.9, 50.0, 72.0)
plot(ecdf(w),verticals = TRUE, do.p = FALSE)
x <- 44:78
lines(x, pnorm(x, mean(w), sd(w)))

# 在R软件中，函数qqnorm()和函数qqline()提供了画正态分布QQ图和相应直线的方法
w <- c(75.0, 64.0, 47.4, 66.9, 62.2, 62.2, 58.7, 63.5, 66.6, 64.0, 57.0, 69.0, 56.9, 50.0, 72.0)
qqnorm(w)
qqline(w)

# 茎叶图、箱线图、及五数概括
x <- c(25, 45, 50, 54, 55, 61, 64, 68, 72, 75, 75, 78, 79, 81, 83, 84, 84, 84, 85, 86, 86, 86,
     87, 89, 89, 89, 90, 91, 91, 92, 100)
stem(x)

# 在茎叶图中，纵轴为测定数据，横轴为数据频数
# 数据的十位数表示“茎”
# 数据的个位数表示“叶”
stem(x, scale=1)
# 如果选择scale=2，那么将数据分为两段，即0-4为一段，5-9为另外一段
stem(x, scale=2)
# 如果选择scale=0.5，那么则有20个数据为一段
stem(x, scale=0.5)

# 箱线图
x <- c(25, 45, 50, 54, 55, 61, 64, 68, 72, 75, 75, 78, 79, 81, 83, 84, 84, 84, 85, 86, 86, 86, 87, 89, 89, 89, 90, 91, 91, 92, 100)
boxplot(x)

# 箱线图比较
A <- c(79.98, 80.04, 80.02, 80.04, 80.03, 80.03, 80.04,
       79.97, 80.05, 80.03, 80.02, 80.00, 80.02)
B <- c(80.02, 79.94, 79.98, 79.97, 79.97, 80.03, 79.95,
       79.97)
boxplot(A, B, notch = T, names=c('A', 'B'), col=c(2,3))

# 没有切口
boxplot(count ~ spray, data = InsectSprays,
        col = "lightgray")

# 有切口
boxplot(count ~ spray, data = InsectSprays,
        notch = TRUE, col = 2:7, add = TRUE)

# 五数总括
# 在探索性分析中，中位数、下四分位数、上四分位数、最小值、最大值
# 这几个数称为样本的五数总括

# 在R软件中，fivenum()函数计算样本的五数总括
x<-c(25, 45, 50, 54, 55, 61, 64, 68, 72, 75, 75,
     78, 79, 81, 83, 84, 84, 84, 85, 86, 86, 86,
     87, 89, 89, 89, 90, 91, 91, 92, 100)
fivenum(x)

# 正态性检验和分布拟合检验
# 正态性W检验方法
# Shapiro-Wilk检验
# 函数shapiro.test()提供W统计量和相应的P值
# 如果P值大于0.05，那么样本为正态分布
w <- c(75.0, 64.0, 47.4, 66.9, 62.2, 62.2, 58.7, 63.5,
       66.6, 64.0, 57.0, 69.0, 56.9, 50.0, 72.0)
shapiro.test(w)

# 随机数的正态性W检验
shapiro.test(runif(100, min = 2, max = 4))


# 经验分布拟合检验的方法
# Kolmogorov-Smirnov方法
# ks.test()函数
x <- rt(100, 5)
x
ks.test(x, "pf", 2,5)

# 高水平的作图函数
# plot()、pairs()、coplot()、qqnorm()、qqline()、hist()、contour()
y<-c(1600, 1610, 1650, 1680, 1700, 1700, 1780, 1500, 1640,
     1400, 1700, 1750, 1640, 1550, 1600, 1620, 1640, 1600,
     1740, 1800, 1510, 1520, 1530, 1570, 1640, 1600)
rep(1,7)
f<-factor(c(rep(1,7),rep(2,5), rep(3,8), rep(4,6)))
plot(f, y)


# 数据框可视化
df<-data.frame(
  Age=c(13, 13, 14, 12, 12, 15, 11, 15, 14, 14, 14,
        15, 12, 13, 12, 16, 12, 11, 15 ),
  Height=c(56.5, 65.3, 64.3, 56.3, 59.8, 66.5, 51.3,
           62.5, 62.8, 69.0, 63.5, 67.0, 57.3, 62.5,
           59.0, 72.0, 64.8, 57.5, 66.5),
  Weight=c( 84.0, 98.0, 90.0, 77.0, 84.5, 112.0,
            50.5, 112.5, 102.5, 112.5, 102.5, 133.0,
            83.0, 84.0, 99.5, 150.0, 128.0, 85.0,
            112.0))

plot(df) # 年龄、身高、体重三项指标的散布图
attach(df)
plot(~Age+Weight)
plot(~Age+Height)
plot(~Weight+Height)

plot(Weight~Age+Height)

# 显示多变量数据
library(bit)
library(bit64)
library(blob)
library(gsubfn)
library(proto)
library(RSQLite)
library(chron)
library(sqldf)

# 显示多变量数据
# pairs()
# coplot()
qqnorm(Age)
qqline(Age)

# 构造数据的点图
dotchart(Age)

# 数据VADeaths给出了弗吉尼亚州的1940年的死亡率
dotchart(VADeaths, main = "Death Rates in Virginia - 1940")
dotchart(t(VADeaths), main = "Death Rates in Virginia - 1940")

# image()绘制出三维图形的印象
# contour()绘制出等值线
# persp()绘制出三维图形的表面曲线

x<-seq(0,2800, 400); y<-seq(0,2400,400)
z<-scan()
1180 1320 1450 1420 1400 1300 700 900
1230 1390 1500 1500 1400 900 1100 1060
1270 1500 1200 1100 1350 1450 1200 1150
1370 1500 1200 1100 1550 1600 1550 1380
1460 1500 1550 1600 1550 1600 1600 1600
1450 1480 1500 1550 1510 1430 1300 1200
1430 1450 1470 1320 1280 1200 1080 940

Z<-matrix(z, nrow=8)
contour(x, y, Z, levels = seq(min(z), max(z), by = 80))
persp(x, y, Z)

persp(x, y, Z, theta = 30, phi = 45, expand = 0.7)

# 高水平绘图中的命令


dotchart(VADeaths, main = "Death Rates in Virginia - 1940")
dotchart(t(VADeaths), main = "Death Rates in Virginia - 1940")

x<-y<-seq(-2*pi, 2*pi, pi/15)
f<-function(x,y) sin(x)*sin(y)
z<-outer(x, y, f)
contour(x,y,z,col="blue")
persp(x,y,z,theta=30, phi=30, expand=0.7,col="lightblue")


# 数据取对数

# log="x" 
# log="y"
# log="xy"

# 多元数据特征和相关性分析
ore<-data.frame(
  x=c(67, 54, 72, 64, 39, 22, 58, 43, 46, 34),
  y=c(24, 15, 23, 19, 16, 11, 20, 16, 17, 13)
)

ore.m <- mean(ore); ore.m
ore.s <- cov(ore); ore.s
ore.r <- cor(ore); ore.r

# mean计算均值
# cov计算计算协方差矩阵
# cor计算相关矩阵

# 二元数据的相关性检验
ruben.test <- function(n, r, alpha=0.05)
{
  u <- qnorm(1-alpha/2)
  r_star <- r/sqrt(1-r^2)
  a <- 2*n-3-u^2
  b <- r_star*sqrt((2*n-3)*(2*n-5))
  c <- (2*n-5-u^2) * r_star^2 - 2*u^2
  y1 <- (b - sqrt(b^2-a*c)) / a
  y2 <- (b + sqrt(b^2-a*c)) / a
  data.frame(n=n, r=r, conf=1-alpha, L=y1/sqrt(1+y1^2), U=y2/sqrt(1+y2^2))
}

ruben.test(6, 0.8)
ruben.test(25, 0.7)

attach(ore)
cor.test(x,y)
cor.test(x,y, method="spearman")
cor.test(x,y, method="kendall")

rubber <- read.table("rubber.data.txt")
rubber
mean(rubber) # 均值
cov(rubber) # 协方差矩阵
cor(rubber) # 相关性矩阵

# 相关性检验
cor.test(~X1+X2, data = rubber)
cor.test(~X1+X3, data = rubber)
cor.test(~X2+X3, data = rubber)

# 基于相关系数的变量分类
rt <- read.table("applicant.txt", header = TRUE)
rt
AVG <- apply(rt, 1, mean)
AVG
sort(AVG, decreasing = TRUE)
order(AVG, decreasing = TRUE)
avgs <- apply(rt, 1, sum)
avgs
sort(avgs, decreasing = TRUE)
order(avgs, decreasing = TRUE)

cor(rt)

attach(rt)
rt$G1 <- (SC+LC+SMS+DRV+AMB+GSP+POT)/7
rt$G2 <- (FL+EXP+SUIT)/3
rt$G3 <- (LA+HON+KJ)/3
rt$G4 <- AA
rt$G5 <- APP

AVG <- apply(rt[,16:20],1,mean)
sort(AVG, decreasing = TRUE)

# 多元数据的图表示方法
# 轮廓图
outline <- function(x, txt = TRUE){
  if (is.data.frame(x) == TRUE)
    x <- as.matrix(x)
  m <- nrow(x); n <- ncol(x)
  plot(c(1,n), c(min(x),max(x)), type = "n",
       main = "The outline graph of Data",
       xlab = "Number", ylab = "Value")
  for(i in 1:m){
    lines(x[i,], col=i)
    if (txt == TRUE){
      k <- dimnames(x)[[1]][i]
      text(1+(i-1)%%n, x[i,1+(i-1)%%n], k)
    }
  }
}

X <- read.table("course.data.txt")
X
outline(X)

# 绘制星图
stars(X)
stars(X, full=FALSE, draw.segments = TRUE,
      key.loc = c(5,0.5), mar = c(2,0,0,0))

# 调和曲线图
unison <- function(x){
  if (is.data.frame(x) == TRUE)
    x <- as.matrix(x)
  t <- seq(-pi, pi, pi/30)
  m <- nrow(x); n<-ncol(x)
  f <- array(0, c(m,length(t)))
  for(i in 1:m){
    f[i,] <- x[i,1]/sqrt(2)
    for( j in 2:n){
      if (j%%2 == 0)
        f[i,] <- f[i,]+x[i,j]*sin(j/2*t)
      else
        f[i,] <- f[i,]+x[i,j]*cos(j%/%2*t)
    }
  }
  plot(c(-pi,pi), c(min(f), max(f)), type = "n",
       main = "The Unison graph of Data",
       xlab = "t", ylab = "f(t)")
  for(i in 1:m) lines(t, f[i,] , col = i)
}

unison(X)



# 第四章：参数估计

moment_fun <- function(p)
{
  f <- c(p[1]*p[2]-A1, p[1]*p[2]-p[1]*p[2]^2-M2)
  J <- matrix(c(p[2],p[1],p[2]-p[2]^2,p[1]-2*p[1]*p[2]), nrow=2, byrow=T)
  list(f=f, J=J)
}

x <- rbinom(100, 20, 0.7)
n <- length(x)
A1 <- mean(x)
M2 <- (n-1)/n*var(x)

Newtons <- function(funs, x, ep=1e-5, it_max=100)
{
  index <- 0
  k <- 1
  while(k<=it_max)
  {
    x1 <- x
    obj <- funs
    x <- x-slove(obj$J, obj$f)
    norm <- sqrt((x-x1) %*% (x-x1))
    if (norm<ep)
    {
      index <- 1
      break
    }
    k <- k+1
  }
  obj <- funs(x)
  list(root=x, it=k, index=index, FunVal=obj$f)
}

p <- c(10,0.5)
Newtons(moment_fun,p)


# 以上代码段有错

# 极大似然法
x <- rcauchy(1000, 1)
f <- function(p) sum((x-p)/(1+(x-p)^2))
out <- uniroot(f, c(0,5))
out

loglike <- function(p) sum(log(1+exp(x-p)^2))
out <- optimize(loglike, c(0, 5))
out


obj<-function(x){
  f<-c(10*(x[2]-x[1]^2), 1-x[1])
  sum(f^2)
}
x0<-c(-1.2,1); nlm(obj,x0)


# 估计量的优良性准则
interval_estimate1<-function(x, sigma=-1, alpha=0.05){
  n<-length(x); xb<-mean(x)
  if (sigma>=0){
    tmp<-sigma/sqrt(n)*qnorm(1-alpha/2); df<-n
  }
  else{
    tmp<-sd(x)/sqrt(n)*qt(1-alpha/2,n-1); df<-n-1
  }
  data.frame(mean=xb, df=df, a=xb-tmp, b=xb+tmp)
}


X<-c(14.6, 15.1,14.9, 14.8, 15.2,15.1)
interval_estimate1(X, sigma = 0.2)
t.test(X)







# 方差sigma的平方

interval_var1<-function(x, mu=Inf, alpha=0.05){
  n<-length(x)
  if (mu<Inf){
    S2 <- sum((x-mu)^2)/n; df <- n
  }
  else{
    S2 <- var(x); df <- n-1
  }
  a<-df*S2/qchisq(1-alpha/2,df)
  b<-df*S2/qchisq(alpha/2,df)
  data.frame(var=S2, df=df, a=a, b=b)
}

X<-c(10.1,10,9.8,10.5,9.7,10.1,9.9,10.2,10.3,9.9)

# 做方差的区间估计，认为均值已知
interval_var1(X, mu=10)

# 做方差的区间估计，认为均值未知
interval_var1(X)





# 两个正态总体的情况
# 均值差mu1-mu2的区间估计



interval_estimate2<-function(x, y,
                             sigma=c(-1,-1), var.equal=FALSE, alpha=0.05){
  n1<-length(x); n2<-length(y)
  xb<-mean(x); yb<-mean(y)
  if (all(sigma>=0)){
    tmp<-qnorm(1-alpha/2)*sqrt(sigma[1]^2/n1+sigma[2]^2/n2)
    df<-n1+n2
  }
  else{
    if (var.equal == TRUE){
      Sw<-((n1-1)*var(x)+(n2-1)*var(y))/(n1+n2-2)
      tmp<-sqrt(Sw*(1/n1+1/n2))*qt(1-alpha/2,n1+n2-2)
      df<-n1+n2-2
    }
    else{
      S1<-var(x); S2<-var(y)
      nu<-(S1/n1+S2/n2)^2/(S1^2/n1^2/(n1-1)+S2^2/n2^2/(n2-1))
      tmp<-qt(1-alpha/2, nu)*sqrt(S1/n1+S2/n2)
      df<-nu
    }
  }
  data.frame(mean=xb-yb, df=df, a=xb-yb-tmp, b=xb-yb+tmp)
}

X <- rnorm(100, 5.32, 2.18)
Y <- rnorm(100, 5.76, 1.76)
interval_estimate2(X, Y, sigma=c(2.18, 1.76))

x <- rnorm(12, 501.1, 2.4)
y <- rnorm(17, 499.7, 4.7)

# 方差相同
interval_estimate2(x, y, var.equal = TRUE)

# 方差不同
interval_estimate2(x, y)

t.test(x, y)
t.test(x, y, var.equal = TRUE)

# 配对数据的区间估计
X<-c(11.3, 15.0, 15.0, 13.5, 12.8, 10.0, 11.0, 12.0, 13.0, 12.3)
Y<-c(14.0, 13.8, 14.0, 13.5, 13.5, 12.0, 14.7, 11.4, 13.8, 12.0)
t.test(X-Y)

# 方差比sigma平方/sigma平方的区间估计

interval_var2<-function(x,y,
                        mu=c(Inf, Inf), alpha=0.05){
  n1<-length(x); n2<-length(y)
  if (all(mu<Inf)){
    Sx2<-1/n1*sum((x-mu[1])^2); Sy2<-1/n2*sum((y-mu[2])^2)
    df1<-n1; df2<-n2
  }
  else{
    Sx2<-var(x); Sy2<-var(y); df1<-n1-1; df2<-n2-1
  }
  r<-Sx2/Sy2
  a<-r/qf(1-alpha/2,df1,df2)
  b<-r/qf(alpha/2,df1,df2)
  data.frame(rate=r, df1=df1, df2=df2, a=a, b=b)
}

A <- c(79.98, 80.04, 80.02, 80.04, 80.03, 80.03, 80.04, 79.97,
       80.05, 80.03, 80.02, 80.00, 80.02)

B<-c(80.02, 79.94, 79.98, 79.97, 79.97, 80.03 ,79.95, 79.97)

interval_var2(A,B,mu=c(80,80))
interval_var2(A,B)

var.test(A,B)



# 非正态总体的区间估计
interval_estimate3<-function(x,sigma=-1,alpha=0.05){
  n<-length(x); xb<-mean(x)
  if (sigma>=0)
    tmp<-sigma/sqrt(n)*qnorm(1-alpha/2)
  else
    tmp<-sd(x)/sqrt(n)*qnorm(1-alpha/2)
  data.frame(mean=xb, a=xb-tmp, b=xb+tmp)
}

x <- rexp(50, 1/2.266)
interval_estimate3(x)

# 单侧置信区间估计
# 一个总体求均值
interval_estimate4<-function(x, sigma=-1, side=0, alpha=0.05){
  n<-length(x); xb<-mean(x)
  if (sigma>=0){
    if (side<0){
      tmp<-sigma/sqrt(n)*qnorm(1-alpha)
      a <- -Inf; b <- xb+tmp
    }
    else if (side>0){
      tmp<-sigma/sqrt(n)*qnorm(1-alpha)
      a <- xb-tmp; b <- Inf
    }
    else{
      tmp <- sigma/sqrt(n)*qnorm(1-alpha/2)
      a <- xb-tmp; b <- xb+tmp
    }
    df<-n
  }
  else{
    if (side<0){
      tmp <- sd(x)/sqrt(n)*qt(1-alpha,n-1)
      a <- -Inf; b <- xb+tmp
    }
    else if (side>0){
      tmp <- sd(x)/sqrt(n)*qt(1-alpha,n-1)
      a <- xb-tmp; b <- Inf
    }
    else{
      tmp <- sd(x)/sqrt(n)*qt(1-alpha/2,n-1)
      a <- xb-tmp; b <- xb+tmp
    }
    df<-n-1
  }
  data.frame(mean=xb, df=df, a=a, b=b)
}

X<-c(1050, 1100, 1120, 1250, 1280)
interval_estimate4(X, side=1)

# R语言中的t.test()函数可以完成单侧区间估计
t.test(X, alternative = "greater")

# 一个总体求方差
interval_var3<-function(x,mu=Inf,side=0,alpha=0.05){
  n<-length(x)
  if (mu<Inf){
    S2<-sum((x-mu)^2)/n; df<-n
  }
  else{
    S2<-var(x); df<-n-1
  }
  if (side<0){
    a <- 0
    b <- df*S2/qchisq(alpha,df)
  }
  else if (side>0){
    a <- df*S2/qchisq(1-alpha,df)
    b <- Inf
  }
  else{
    a<-df*S2/qchisq(1-alpha/2,df)
    b<-df*S2/qchisq(alpha/2,df)
  }
  data.frame(var=S2, df=df, a=a, b=b)
}

X<-c(10.1,10,9.8,10.5,9.7,10.1,9.9,10.2,10.3,9.9)
interval_var3(X, side=-1)




interval_estimate5<-function(x, y,
                             sigma=c(-1,-1), var.equal=FALSE, side=0, alpha=0.05){
  n1<-length(x); n2<-length(y)
  xb<-mean(x); yb<-mean(y); zb<-xb-yb
  if (all(sigma>=0)){
    if (side<0){
      tmp<-qnorm(1-alpha)*sqrt(sigma[1]^2/n1+sigma[2]^2/n2)
      a <- -Inf; b <- zb+tmp
    }
    else if (side>0){
      tmp<-qnorm(1-alpha)*sqrt(sigma[1]^2/n1+sigma[2]^2/n2)
      a <- zb-tmp; b <- Inf
    }
    else{
      tmp<-qnorm(1-alpha/2)*sqrt(sigma[1]^2/n1+sigma[2]^2/n2)
      a <- zb-tmp; b <- zb+tmp
    }
    df<-n1+n2
  }
  else{
    if (var.equal == TRUE){
      Sw<-((n1-1)*var(x)+(n2-1)*var(y))/(n1+n2-2)
      if (side<0){
        tmp<-sqrt(Sw*(1/n1+1/n2))*qt(1-alpha,n1+n2-2)
        a <- -Inf; b <- zb+tmp
      }
      else if (side>0){
        tmp<-sqrt(Sw*(1/n1+1/n2))*qt(1-alpha,n1+n2-2)
        a <- zb-tmp; b <- Inf
      }
      else{
        tmp<-sqrt(Sw*(1/n1+1/n2))*qt(1-alpha/2,n1+n2-2)
        a <- zb-tmp; b <- zb+tmp
      }
      df<-n1+n2-2
    }
    else{
      S1<-var(x); S2<-var(y)
      nu<-(S1/n1+S2/n2)^2/(S1^2/n1^2/(n1-1)+S2^2/n2^2/(n2-1))
      if (side<0){
        tmp<-qt(1-alpha, nu)*sqrt(S1/n1+S2/n2)
        a <- -Inf; b <- zb+tmp
      }
      else if (side>0){
        tmp<-qt(1-alpha, nu)*sqrt(S1/n1+S2/n2)
        a <- zb-tmp; b <- Inf
      }
      else{
        tmp<-qt(1-alpha/2, nu)*sqrt(S1/n1+S2/n2)
        a <- zb-tmp; b <- zb+tmp
      }
      df<-nu
    }
  }
  data.frame(mean=zb, df=df, a=a, b=b)
}


# 求两个总体方差情况
interval_var4<-function(x,y,
                        mu=c(Inf, Inf), side=0, alpha=0.05){
  n1<-length(x); n2<-length(y)
  if (all(mu<Inf)) {
    Sx2<-1/n1*sum((x-mu[1])^2); df1<-n1
    Sy2<-1/n2*sum((y-mu[2])^2); df2<-n2
  }
  else{
    Sx2<-var(x); Sy2<-var(y); df1<-n1-1; df2<-n2-1
  }
  r<-Sx2/Sy2
  if (side<0) {
    a <- 0
    b <- r/qf(alpha,df1,df2)
  }
  else if (side>0) {
    a <- r/qf(1-alpha,df1,df2)
    b <- Inf
  }
  else{
    a<-r/qf(1-alpha/2,df1,df2)
    b<-r/qf(alpha/2,df1,df2)
  }
  data.frame(rate=r, df1=df1, df2=df2, a=a, b=b)
}




# 第五章 假设检验
# 重要的参数检验
# 正态总体均值的假设检验
# 单个总体的情况
P_value<-function(cdf, x, paramet=numeric(0), side=0){
  n<-length(paramet)
  P<-switch(n+1,
            cdf(x),
            cdf(x, paramet),
            cdf(x, paramet[1], paramet[2]),
            cdf(x, paramet[1], paramet[2], paramet[3])
  )
  if (side<0) P
  else if (side>0) 1-P
  else
    if (P<1/2) 2*P
  else 2*(1-P)
}

mean.test1<-function(x, mu=0, sigma=-1, side=0){
  
  n<-length(x); xb<-mean(x)
  if (sigma>0){
    z<-(xb-mu)/(sigma/sqrt(n))
    P<-P_value(pnorm, z, side=side)
    data.frame(mean=xb, df=n, Z=z, P_value=P)
  }
  else{
    t<-(xb-mu)/(sd(x)/sqrt(n))
    P<-P_value(pt, t, paramet=n-1, side=side)
    data.frame(mean=xb, df=n-1, T=t, P_value=P)
  }
}


X<-c(159, 280, 101, 212, 224, 379, 179, 264,
     222, 362, 168, 250, 149, 260, 485, 170)
mean.test1(X, mu=225, side=1)
interval_estimate4(X, side=1)
t.test(X, alternative = "greater", mu = 225)

# 两个总体的情况
mean.test2<-function(x, y,
                     sigma=c(-1, -1), var.equal=FALSE, side=0){
  
  n1<-length(x); n2<-length(y)
  xb<-mean(x); yb<-mean(y)
  if (all(sigma>0)){
    z<-(xb-yb)/sqrt(sigma[1]^2/n1+sigma[2]^2/n2)
    P<-P_value(pnorm, z, side=side)
    data.frame(mean=xb-yb, df=n1+n2, Z=z, P_value=P)
  }
  else{
    if (var.equal == TRUE){
      Sw<-sqrt(((n1-1)*var(x)+(n2-1)*var(y))/(n1+n2-2))
      t<-(xb-yb)/(Sw*sqrt(1/n1+1/n2))
      nu<-n1+n2-2
    }
    else{
      S1<-var(x); S2<-var(y)
      nu<-(S1/n1+S2/n2)^2/(S1^2/n1^2/(n1-1)+S2^2/n2^2/(n2-1))
      t<-(xb-yb)/sqrt(S1/n1+S2/n2)
    }
    P<-P_value(pt, t, paramet=nu, side=side)
    data.frame(mean=xb-yb, df=nu, T=t, P_value=P)
  }
}

X<-c(78.1,72.4,76.2,74.3,77.4,78.4,76.0,75.5,76.7,77.3)
Y<-c(79.1,81.0,77.3,79.1,80.0,79.1,79.1,77.3,80.2,82.1)
# 两方差相同
mean.test2(X, Y, var.equal=TRUE, side=-1)
# 两方差不同
mean.test2(X, Y, side=-1)

# 做单侧区间估计，并且认为两总体方差相同
interval_estimate5(X, Y, var.equal=TRUE, side=-1)

# 做单侧区间估计，并且认为两总体方差不同
interval_estimate5(X,Y, side=-1)

# t.test()也可以做双样本检验
t.test(X, Y, var.equal=TRUE, alternative = "less")

# 成对数据的T检验
X<-c(78.1,72.4,76.2,74.3,77.4,78.4,76.0,75.5,76.7,77.3)
Y<-c(79.1,81.0,77.3,79.1,80.0,79.1,79.1,77.3,80.2,82.1)
t.test(X-Y, alternative = "less")


# 正态总体方差的假设检验
# 单个总体情况
var.test1<-function(x, sigma2=1, mu=Inf, side=0){
  
  n<-length(x)
  if (mu<Inf){
    S2<-sum((x-mu)^2)/n; df=n
  }
  else{
    S2<-var(x); df=n-1
  }
  chi2<-df*S2/sigma2;
  P<-P_value(pchisq, chi2, paramet=df, side=side)
  data.frame(var=S2, df=df, chisq2=chi2, P_value=P)
}

X<-c(136, 144 ,143 ,157 ,137 ,159 ,135 ,158 ,147 ,165,
158 ,142 ,159 ,150 ,156 ,152 ,140 ,149 ,148 ,155)

# 认为方差已知，做均值检验
mean.test1(X, mu=149, sigma = sqrt(75))

# 认为方差未知，做均值检验
mean.test1(X, mu=149)

# 调用var.test1函数
# 认为均值已知，做方差检验
var.test1(X, sigma2=75, mu=149)

# 认为均值未知，做方差检验
var.test1(X, sigma2=75)

# 两个总体的情况
var.test2<-function(x, y, mu=c(Inf, Inf), side=0){
  
  n1<-length(x); n2<-length(y)
  if (all(mu<Inf)){
    Sx2<-sum((x-mu[1])^2)/n1; Sy2<-sum((y-mu[2])^2)/n2
    df1=n1; df2=n2
  }
  else{
    Sx2<-var(x); Sy2<-var(y); df1=n1-1; df2=n2-1
  }
  r<-Sx2/Sy2
  P<-P_value(pf, r, paramet=c(df1, df2), side=side)
  data.frame(rate=r, df1=df1, df2=df2, F=r, P_value=P)
}

X<-c(78.1,72.4,76.2,74.3,77.4,78.4,76.0,75.5,76.7,77.3)
Y<-c(79.1,81.0,77.3,79.1,80.0,79.1,79.1,77.3,80.2,82.1)
var.test2(X, Y)
# 做方差比的区间估计，考虑均值未知的状况
interval_var4(X, Y)
var.test(X,Y)

# 二项分布的总体的假设检验
# 使用binom.test()函数
binom.test(445,500,p=0.85)

binom.test(1, 400, p = 0.01, alternative = "less")
binom.test(c(1, 399), p = 0.01, alternative = "less")


# 若干重要的非参数检验
# Pearson拟合优度卡方检验
# 理论分布完全已知的情况
X<-c(210, 312, 170, 85, 223)
n<-sum(X); m<-length(X)
p<-rep(1/m, m)
K<-sum((X-n*p)^2/(n*p));K
Pr<-1-pchisq(K, m-1);Pr

chisq.test(X)

# 第一步，输入数据
X <- scan()
25 45 50 54 55 61 64 68 72 75 75
78 79 81 83 84 84 84 85 86 86 86
87 89 89 89 90 91 91 92 100
# 第二步，分组和计数
A <- table(cut(X, br=c(0,69,79,89,100)))
# 第三步，构造理论分布
p <- pnorm(c(70,80,90,100),mean(X), sd(X))
# 第四步，作检验
chisq.test(A, p=p)

chisq.test(c(335,125,160), p=c(9,3,4)/16)

# 输入数据
X <- 0:6
Y <- c(7,10,12,8,3,2,0)
# 计算理论分布
q <- ppois(X, mean(rep(X,Y)))
n <- length(Y)
p[1] <- q[1]
p[n] <- 1-q[n-1]
for (i in 2:(n-1))
{
  p[i] <- q[i] - q[i-1]
}

# 做卡方检验
chisq.test(Y, p=p)

#### 
Z<-c(7, 10, 12, 8, 5)
#### 
n<-length(Z); p<-p[1:n-1]; p[n]<-1-q[n-1]
####
chisq.test(Z, p=p)


# Kolmogorov-Smirnov检验
# 单样本检验
X<-c(420, 500, 920, 1380, 1510, 1650, 1760, 2100, 2300, 2350)
ks.test(X, "pexp", 1/1500)

# 双样本检验
X <- c(0.61, 0.29, 0.06, 0.59, -1.73, -0.74, 0.51, -0.56, 0.39,
       1.64, 0.05, -0.06, 0.64, -0.82, 0.37, 1.77, 1.09, -1.28,
       2.36, 1.31, 1.05, -0.32, -0.40, 1.06, -2.47)

Y <- c(2.20, 1.66, 1.38, 0.20, 0.36, 0.00, 0.96, 1.56, 0.44,
       1.50, -0.30, 0.66, 2.31, 3.29, -0.27, -0.37, 0.38, 0.70,
       0.52, -0.71)

# 做KS检验
ks.test(X,Y)

# 列联表的独立性检验
# chisq.test()函数也可以做独立性检验
# 只需要将列联表数据转化为矩阵即可
x <- c(60, 2, 32, 11)
dim(x) <- c(2,2)
chisq.test(x, correct = FALSE)
# 或者带联续校正
chisq.test(x)

x <- c(20,24,80,82,22,38,104,125,13,28,81,113,7,18,54,92)
dim(x) <- c(4,4)
chisq.test(x)

# 如果卡方检验是合理的
# 那么使用Fisher精确检验
# Fisher精确独立检验
# 函数fisher.test()作精确概率检验
x <- c(4,5,18,6)
dim(x) <- c(2,2)
fisher.test(x)

x <- c(60,3,32,11)
dim(x) <- c(2,2)
fisher.test(x)

# McNemar检验
# McNemar检验是在相同个体上的两次检验
# 检验两个数据的两个相关分布的频数比变化的显著性
# 使用mcnemar.test()检验
x <- c(49,21,25,107)
dim(x) <- c(2,2)
mcnemar.test(x, correct = FALSE)

# 符号检验
# 检验一个样本是否来自总体
X<-scan()
66 75 78 80 81 81 82 83 83 83 83
84 85 85 86 86 86 86 87 87 88 88
88 88 88 89 89 89 89 90 90 91 91
91 91 92 93 93 96 96 96 97 99 100
101 102 103 103 104 104 104 105 106 109 109
110 110 110 111 113 115 116 117 118 155 192

# 开始检验
binom.test(sum(X > 99), length(X), al='l')

# 用成对样本来检查两个总体间是否存在显著性差异
x <- c(25, 30, 28, 23, 27, 35, 30, 28, 32, 29, 30, 30, 31, 16)

y <- c(19, 32, 21, 19, 25, 31, 31, 26, 30, 25, 28, 31, 25, 25)

binom.test(sum(x<y), length(x))

# 单边备择假设检验
binom.test(3,12,p=1/2,al='l',conf.level = 0.90)

# 秩统计量
# 秩检验统计量非参数检验中广泛有用的统计量
# 重要的特性是分布无关性
x <- c(1.2,0.8,-3.1,2.0,1.2)
rank(x)
x <- c(1.2, 0.8, -3.1, 2.0, 1.2+1e-5)
rank(x)

# 秩相关检验
# Spearman秩相关检验
x <- c(1,2,3,4,5,6)
y <- c(6,5,4,3,2,1)
cor.test(x,y,method = "spearman")

# Kendall相关检验
X<-c(86, 77, 68, 91, 70, 71, 85, 87, 63)
Y<-c(88, 76, 64, 96, 65, 80, 81, 72, 60)
cor.test(X, Y, method = "kendall")

# Wilcoxon秩检验
X <- c(137.0, 140.0, 138.3, 139.0, 144.3, 139.1, 141.7, 137.3, 133.5, 138.2,
       141.1, 139.2, 136.5, 136.5, 135.6, 138.0, 140.9, 140.6, 136.3, 134.1)

wilcox.test(X, mu=140, alternative="less",
            exact=FALSE, correct=FALSE, conf.int=TRUE)

x <- c(24,26,29,34,43,58,63,72,87,101)



# 成对样本的检验
x <- c(459,367,303,392,310,342,421,446,430,412)
y <- c(414,306,321,443,281,301,353,391,405,390)
wilcox.test(x, y, alternative = "greater", paired = TRUE)
# 一下有相同效果
wilcox.test(x-y, alternative = "greater")

# 如符号检验计算
binom.test(sum(x>y), length(x), alternative = "greater")

# 非成对样本的秩次和检验
x <- c(24,26,29,34,43,58,63,72,87,101)
y <- c(82,87,97,121,164,208,213)
# 不采取连续修正
wilcox.test(x, y, alternative = "less", exact = FALSE, correct = FALSE)
# 采取连续修正
wilcox.test(x, y, alternative = "less", exact = FALSE)

# W=4.5 是Wilcox-Mann-Whitney统计量

x <- c(3,5,7,9,10)
y <- c(1,2,4,6,8)
wilcox.test(x, y, alternative = "greater")

X <- c(4,6,7,9,10)
Y <- c(1,2,3,5,8)
wilcox.test(X, Y, alternative = "greater")

# 各个病人的疗效用4个不同的值表示
x <- rep(1:4, c(62,41,14,11))
y <- rep(1:4, c(20,37,16,15))
wilcox.test(x, y, exact = FALSE)

# 一元线性回归
x <- c(0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
       0.17, 0.18, 0.20, 0.21, 0.23)
y <- c(42.0, 43.5, 45.0, 45.5, 45.0, 47.5, 49.0, 
       53.0, 50.0, 55.0, 55.0, 60.0)
lm.sol <- lm(y~1+x)
summary(lm.sol)

# y ~ 1+x 表示常数项+一次项+残差

# summary表示展示一元线性回归的结果

# Call 表示模型的公式
# Residuals 表示列出的是残差值的：
# 最小值、1/4分位数、中位数、3/4分位数、最大值
# Coefficients 表示：
#   Estimate 表示回归方程的参数估计
#   Std. Error 表示回归方程的参数的标准差
#   t value 表示回归方程参数的t值
#   Pr(>|t|) 表示P值
#     还有显著性标记
#     *** 表示 极为显著
#     ** 表示 高度显著
#     * 表示 显著
#     . 表示 不太显著
#     没有记号 表示 不显著
# Residual standard error 表示残差的标准差
# Multiple R-squared 表示 相关系数的平方
# F-statistic 表示 F统计检验量
# p-value 表示P值

# 参数beta0和beta1的区间估计
beta.int <- function(fm, alpha=0.5)
{
  A <- summary(fm)$coefficients
  df <- fm$df.residual
  left <- A[,1]-A[,2]*qt(1-alpha/2,df)
  right <- A[,1]+A[,2]*qt(1-alpha/2,df)
  rowname <- dimnames(A)[[1]]
  colname <- c("Estimate", "Left", "Right")
  matrix(c(A[,1], left, right), ncol = 3,
         dimnames=list(rowname, colname))
}

# 得到相应的模型估计
beta.int(lm.sol)

# 预测
new <- data.frame(x=0.16) # 注意，即使只有一个新点，还是要采用数据框的形式
lm.pred <- predict(lm.sol, new, interval = "prediction", level = 0.95)
# interval = "prediction" 表示要有预测区间
lm.pred

# 没有预测区间
lm.pred_nainterval <- predict(lm.sol, new, level = 0.95)
lm.pred_nainterval

# Forbes数据示例
X <- matrix(c(
  194.5, 20.79, 1.3179, 131.79,
  194.3, 20.79, 1.3179, 131.79,
  197.9, 22.40, 1.3502, 135.02,
  198.4, 22.67, 1.3555, 135.55,
  199.4, 23.15, 1.3646, 136.46,
  199.9, 23.35, 1.3683, 136.83,
  200.9, 23.89, 1.3782, 137.82,
  201.1, 23.99, 1.3800, 138.00,
  201.4, 24.02, 1.3806, 138.06,
  201.3, 24.01, 1.3805, 138.05,
  203.6, 25.14, 1.4004, 140.04,
  204.6, 26.57, 1.4244, 142.44,
  209.5, 28.49, 1.4547, 145.47,
  208.6, 27.76, 1.4434, 144.34,
  210.7, 29.04, 1.4630, 146.30,
  211.9, 29.88, 1.4754, 147.54,
  212.2, 30.06, 1.4780, 147.80),
  ncol=4, byrow=T,
  dimnames = list(1:17, c("F", "h", "log", "log100")))

X

# 转化为数据框
forbes <- as.data.frame(X)
forbes

# 数据相关关系 图示
plot(forbes$F, forbes$log100)

# 回归
lm.huigui <- lm(log100~F, data = forbes)

summary(lm.huigui)

# 绘制回归图线
abline(lm.huigui)

# 计算残差
# residuals()函数计算
# 并且绘制出残差图
y.res <- residuals(lm.huigui)
plot(y.res)
text(12, y.res[12], labels = 12, adj = 1.2)

# 做一个检查
# 在数据中，去掉12号点
i <- 1:17
forbes12 <- as.data.frame(X[i!=12,])
lm12 <- lm(log100~F ,data=forbes12)
summary(lm12)

plot(residuals(lm12))

# R语言中与线性模型有关的函数

# 基本函数
# lm()

# 提取模型通用信息的函数
# lm()函数的返回值的本质是一个 具有类属性值lm的列表
# add1
# coef 提取模型系数
# effects
# kappa
# predict 做预测
# residuals 计算残差
# alias
# deviance 计算残差平方和
# family
# labels
# print 显示
# step 做逐步的回归
# anova 计算方差分析表
# drop1
# formula 提取模型公式
# plot 绘制模型诊断图
# proj 
# summary 提取模型资料


# 多元线性回归模型
blood<-data.frame(
  X1=c(76.0, 91.5, 85.5, 82.5, 79.0, 80.5, 74.5,
       79.0, 85.0, 76.5, 82.0, 95.0, 92.5),
  X2=c(50, 20, 20, 30, 30, 50, 60, 50, 40, 55,
       40, 40, 20),
  Y= c(120, 141, 124, 126, 117, 125, 123, 125,
       132, 123, 132, 155, 147)
)

blood
lm.sol <- lm(Y ~ X1+X2, data = blood)
summary(lm.sol)

# 参数区间估计
beta.int <- function(fm, alpha=0.5)
{
  A <- summary(fm)$coefficients
  df <- fm$df.residual
  left <- A[,1]-A[,2]*qt(1-alpha/2,df)
  right <- A[,1]+A[,2]*qt(1-alpha/2,df)
  rowname <- dimnames(A)[[1]]
  colname <- c("Estimate", "Left", "Right")
  matrix(c(A[,1], left, right), ncol = 3,
         dimnames=list(rowname, colname))
}

beta.int(lm.sol)

# 预测
new <- data.frame(X1=80, X2=40)
lm.pred <- predict(lm.sol, new, interval = "prediction", level = 0.95)
lm.pred

# 修正拟合模型
# update()是一个非常方便修正模型的函数
toothpaste<-data.frame(
  X1=c(-0.05, 0.25,0.60,0, 0.25,0.20, 0.15,0.05,-0.15, 0.15,
       0.20, 0.10,0.40,0.45,0.35,0.30, 0.50,0.50, 0.40,-0.05,
       -0.05,-0.10,0.20,0.10,0.50,0.60,-0.05,0, 0.05, 0.55),
  X2=c( 5.50,6.75,7.25,5.50,7.00,6.50,6.75,5.25,5.25,6.00,
        6.50,6.25,7.00,6.90,6.80,6.80,7.10,7.00,6.80,6.50,
        6.25,6.00,6.50,7.00,6.80,6.80,6.50,5.75,5.80,6.80),
  Y =c( 7.38,8.51,9.52,7.50,9.33,8.28,8.75,7.87,7.10,8.00,
        7.89,8.15,9.10,8.86,8.90,8.87,9.26,9.00,8.75,7.95,
        7.65,7.27,8.00,8.50,8.75,9.21,8.27,7.67,7.93,9.26)
)
# 展示
toothpaste

# 进行建模
lm.sol <- lm(Y~X1+X2, data = toothpaste)
summary(lm.sol)

# 绘制x1与y的散点图和回归直线
attach(toothpaste)
plot(Y~X1)
abline(lm(Y~X1))

# 绘制x2与y的散点图和回归曲线
lm2.sol <- lm(Y~X2+I(X2^2))
summary(lm2.sol)
x <- seq(min(X2), max(X2), len=200)
x
y <- predict(lm2.sol, data.frame(X2=x))
y

plot(Y~X2)
lines(x,y)

# 做相应的回归分析
lm.new <- update(lm.sol, .~.+I(X2^2))
summary(lm.new)
beta.int(lm.new)


# 去掉X2的一次项
lm2.new <- update(lm.new, .~.-X2)
summary(lm2.new)


# 考虑x1和x2的交互作用
lm3.new <- update(lm.new, .~.+X1*X2)
summary(lm3.new)


# 逐步回归
cement <- data.frame(
  X1=c( 7, 1, 11, 11, 7, 11, 3, 1, 2, 21, 1, 11, 10),
  X2=c(26, 29, 56, 31, 52, 55, 71, 31, 54, 47, 40, 66, 68),
  X3=c( 6, 15, 8, 8, 6, 9, 17, 22, 18, 4, 23, 9, 8),
  X4=c(60, 52, 20, 47, 33, 22, 6, 44, 22, 26, 34, 12, 12),
  Y =c(78.5, 74.3, 104.3, 87.6, 95.9, 109.2, 102.7, 72.5,
       93.1,115.9, 83.8, 113.3, 109.4)
)
cement

lm.sol <- lm(Y ~ X1+X2+X3+X4, data = cement)
summary(lm.sol)

# 尝试逐步回归
lm.step <- step(lm.sol)
summary(lm.step)

# 做逐步回归的函数
drop1(lm.step)

# 再去掉一个指标
lm.opt <- lm(Y ~ X1+X2, data = cement)
summary(lm.opt)

# 回归诊断
# 图的有用性
Anscombe<-data.frame(
  X=c(10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0),
  Y1=c(8.04,6.95, 7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68),
  Y2=c(9.14,8.14, 8.74,8.77,9.26,8.10,6.13,3.10, 9.13,7.26,4.74),
  Y3=c(7.46,6.77,12.74,7.11,7.81,8.84,6.08,5.39, 8.15,6.44,5.73),
  X4=c(rep(8,7), 19, rep(8,3)),
  Y4=c(6.58,5.76,7.71,8.84,8.47,7.04,5.25,12.50, 5.56,7.91,6.89)
)

Anscombe

summary(lm(Y1~X, data = Anscombe))
summary(lm(Y2~X, data = Anscombe))
summary(lm(Y3~X, data = Anscombe))
summary(lm(Y4~X4,data = Anscombe))

lm2.sol <- lm(Y2~X+I(X^2), data = Anscombe)
summary(lm2.sol)

# 以下函数与回归诊断有关
# influence.measures 
# rstandard 
# rstudent 
# dffits
# cooks.distance 
# dfbeta dfbetas 
# covratio
# hatvalues 
# hat



# 例子6.5
# Forbes数据示例
X <- matrix(c(
  194.5, 20.79, 1.3179, 131.79,
  194.3, 20.79, 1.3179, 131.79,
  197.9, 22.40, 1.3502, 135.02,
  198.4, 22.67, 1.3555, 135.55,
  199.4, 23.15, 1.3646, 136.46,
  199.9, 23.35, 1.3683, 136.83,
  200.9, 23.89, 1.3782, 137.82,
  201.1, 23.99, 1.3800, 138.00,
  201.4, 24.02, 1.3806, 138.06,
  201.3, 24.01, 1.3805, 138.05,
  203.6, 25.14, 1.4004, 140.04,
  204.6, 26.57, 1.4244, 142.44,
  209.5, 28.49, 1.4547, 145.47,
  208.6, 27.76, 1.4434, 144.34,
  210.7, 29.04, 1.4630, 146.30,
  211.9, 29.88, 1.4754, 147.54,
  212.2, 30.06, 1.4780, 147.80),
  ncol=4, byrow=T,
  dimnames = list(1:17, c("F", "h", "log", "log100")))

X

# 转化为数据框
forbes <- as.data.frame(X)
forbes

# 数据相关关系 图示
plot(forbes$F, forbes$log100)

# 回归
lm.huigui <- lm(log100~F, data = forbes)

summary(lm.huigui)

# 绘制回归图线
abline(lm.huigui)

# 计算残差
# residuals()函数计算
# 并且绘制出残差图
y.res <- residuals(lm.huigui)
plot(y.res)
text(12, y.res[12], labels = 12, adj = 1.2)

# 计算残差，做残差的正态性检验
y.res <- residuals(lm.huigui)
shapiro.test(y.res)
# 由于P值小于0.05，不满足正态性的条件


# 标准化残差
# 内学生化残差
# 函数rstandard()计算标准化残差






# 外学生化残差
# 函数rstudent()计算外学生化残差


# 例子6.5

# 画残差图
y.res <- resid(lm.huigui)
y.fit <- predict(lm.huigui)
plot(y.res~y.fit)

# 画标准化残差图
y.rst <- rstandard(lm.huigui)
plot(y.rst~y.fit)

# 例子6.14
X<-scan()
679 292 1012 493 582 1156 997 2189 1097 2078
1818 1700 747 2030 1643 414 354 1276 745 435
540 874 1543 1029 710 1434 837 1748 1381 1428
1255 1777 370 2316 1130 463 770 724 808 790
783 406 1242 658 1746 468 1114 413 1787 3560
1495 2221 1526

# 此处在控制台上敲个回车

# 前面需要清除


Y<-scan()
0.79 0.44 0.56 0.79 2.70 3.64 4.73 9.50 5.34 6.85
5.84 5.21 3.25 4.43 3.16 0.50 0.17 1.88 0.77 1.39
0.56 1.56 5.28 0.64 4.00 0.31 4.20 4.88 3.48 7.58
2.63 4.99 0.59 8.19 4.79 0.51 1.74 4.10 3.94 0.96
3.29 0.44 3.24 2.14 5.71 0.64 1.90 0.51 8.33 14.94
5.11 3.85 3.93

# 此处在控制台上敲个回车

lm.sol <- lm(Y~X)
summary(lm.sol)

# 再做回归诊断
# 画出残差化标准散点图
y.rst <- rstandard(lm.sol)
y.fit <- predict(lm.sol)
plot(y.rst~y.fit)
abline(0.1,0.5)
abline(-0.1,-0.5)


# 异方差情景
lm.new <- update(lm.sol, sqrt(.)~.)
coef(lm.new)

# 再绘制出新模型的标准化残差散点图
yn.rst <- rstandard(lm.new)
yn.fit <- predict(lm.new)
plot(yn.rst~yn.fit)

abline(0.1,0.5)
abline(-0.1,-0.5)


# 残差的QQ图


# 以自变量为横坐标的残差图
y.res <- resid(lm.sol)
plot(y.res~X)

# 影响分析
# 帽子矩阵H的对角元素

# DFFITS准则
X <- matrix(c(
  194.5, 20.79, 1.3179, 131.79,
  194.3, 20.79, 1.3179, 131.79,
  197.9, 22.40, 1.3502, 135.02,
  198.4, 22.67, 1.3555, 135.55,
  199.4, 23.15, 1.3646, 136.46,
  199.9, 23.35, 1.3683, 136.83,
  200.9, 23.89, 1.3782, 137.82,
  201.1, 23.99, 1.3800, 138.00,
  201.4, 24.02, 1.3806, 138.06,
  201.3, 24.01, 1.3805, 138.05,
  203.6, 25.14, 1.4004, 140.04,
  204.6, 26.57, 1.4244, 142.44,
  209.5, 28.49, 1.4547, 145.47,
  208.6, 27.76, 1.4434, 144.34,
  210.7, 29.04, 1.4630, 146.30,
  211.9, 29.88, 1.4754, 147.54,
  212.2, 30.06, 1.4780, 147.80),
  ncol=4, byrow=T,
  dimnames = list(1:17, c("F", "h", "log", "log100")))

X

# 转化为数据框
forbes <- as.data.frame(X)
lm.sol <- lm(log100~F, data=forbes)
p <- 1
n <- nrow(forbes)
d <- dffits(lm.sol)
cf <- 1:n
cf[d>2*sqrt((p+1)/n)]

# cook统计量

# covratio准则

# 小结
# 编写回归诊断函数
# Reg_Diag()
Reg_Diag<-function(fm){
  n<-nrow(fm$model); df<-fm$df.residual
  p<-n-df-1; s<-rep(" ", n);
  res<-residuals(fm); s1<-s; s1[abs(res)==max(abs(res))]<-"*"
  sta<-rstandard(fm); s2<-s; s2[abs(sta)>2]<-"*"
  stu<-rstudent(fm); s3<-s; s3[abs(sta)>2]<-"*"
  h<-hatvalues(fm); s4<-s; s4[h>2*(p+1)/n]<-"*"
  d<-dffits(fm); s5<-s; s5[abs(d)>2*sqrt((p+1)/n)]<-"*"
  c<-cooks.distance(fm); s6<-s; s6[c==max(c)]<-"*"
  co<-covratio(fm); abs_co<-abs(co-1)
  s7<-s; s7[abs_co==max(abs_co)]<-"*"
  data.frame(residual=res, s1, standard=sta, s2,
             student=stu, s3, hat_matrix=h, s4,
             DFFITS=d, s5,cooks_distance=c, s6,
             COVRATIO=co, s7)
}

# 在给定回归模型之后，计算回归模型的普通残差、标准化残差、外学生化残差、帽子矩阵对角线上的元素、
# DFFITS统计量、Cook距离、covratio统计量

Reg_Diag(lm.sol)

# 例子 6.17
intellect<-data.frame(
  x=c(15, 26, 10, 9, 15, 20, 18, 11, 8, 20, 7,
      9, 10, 11, 11, 10, 12, 42, 17, 11, 10),
  y=c(95, 71, 83, 91, 102, 87, 93, 100, 104, 94, 113,
      96, 83, 84, 102, 100, 105, 57, 121, 86, 100)
)

lm.sol <- lm(y~x, data=intellect)
summary(lm.sol)

# 做回归诊断
Reg_Diag(lm.sol)
# 通过回归诊断，判断19是异常值

# 设置画图选项
opar <- par(mfrow = c(2, 2), oma = c(0, 0, 1.1, 0),
            mar = c(4.1, 4.1, 2.1, 1.1))
plot(lm.sol, 1)
plot(lm.sol, 3)
plot(lm.sol, 4)
attach(intellect)
plot(x,y)
X <- x[18:19]
Y <- y[18:19]
test(X, Y, labels=18:19, adj=1.2)
abline(lm.sol)
par(opar)

# 多重共线性
# 条件数k
# 当k<100，认为多重共线性的值很小
# 当100<=k<=1000，认为存在中等或者较强的多重共线性
# 当1000<k，认为存在严重的多重共线性
collinear<-data.frame(
  Y=c(10.006, 9.737, 15.087, 8.422, 8.625, 16.289,
      5.958, 9.313, 12.960, 5.541, 8.756, 10.937),
  X1=rep(c(8, 0, 2, 0), c(3, 3, 3, 3)),
  X2=rep(c(1, 0, 7, 0), c(3, 3, 3, 3)),
  X3=rep(c(1, 9, 0), c(3, 3, 6)),
  X4=rep(c(1, 0, 1, 10), c(1, 2, 6, 3)),
  X5=c(0.541, 0.130, 2.116, -2.397, -0.046, 0.365,
       1.996, 0.228, 1.38, -0.798, 0.257, 0.440),
  X6=c(-0.099, 0.070, 0.115, 0.252, 0.017, 1.504,
       -0.865, -0.055, 0.502, -0.399, 0.101, 0.432)
)


XX <- cor(collinear[2:7])

kappa(XX, exact = TRUE)

# 找出哪些是多重共线性的
eigen(XX)


# 广义线性回归
# 函数glm()
norell<-data.frame(
  x=0:5, n=rep(70,6), success=c(0,9,21,47,60,63)
)
norell
# x为电流
# n为实验次数
# success为响应次数

norell$Ymat <- cbind(norell$success, norell$n-norell$success)
norell
norell$Ymat <- cbind(norell$success, norell$n-norell$success)
glm.sol<-glm(Ymat~x, family=binomial, data=norell)
summary(glm.sol)

pre<-predict(glm.sol, data.frame(x=3.5))
p<-exp(pre)/(1+exp(pre)); p

X<- - glm.sol$coefficients[[1]]/glm.sol$coefficients[[2]]
X

d<-seq(0, 5, len=100)
pre<-predict(glm.sol, data.frame(x = d))
p<-exp(pre)/(1+exp(pre))
norell$y<-norell$success/norell$n
plot(norell$x, norell$y); lines(d, p)


# 例子6.20
# 试用逻辑回归分析病人生存时间长短的概率与X1、X2、X3的关系
life<-data.frame(
  X1=c(2.5, 173, 119, 10, 502, 4, 14.4, 2, 40, 6.6,
       21.4, 2.8, 2.5, 6, 3.5, 62.2, 10.8, 21.6, 2, 3.4,
       5.1, 2.4, 1.7, 1.1, 12.8, 1.2, 3.5, 39.7, 62.4, 2.4,
       34.7, 28.4, 0.9, 30.6, 5.8, 6.1, 2.7, 4.7, 128, 35,
       2, 8.5, 2, 2, 4.3, 244.8, 4, 5.1, 32, 1.4),
  X2=rep(c(0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
           0, 2, 0, 2, 0, 2, 0),
         c(1, 4, 2, 2, 1, 1, 8, 1, 5, 1, 5, 1, 1, 1, 2, 1,
           1, 1, 3, 1, 2, 1, 4)),
  X3=rep(c(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
         c(6, 1, 3, 1, 3, 1, 1, 5, 1, 3, 7, 1, 1, 3, 1, 1, 2, 9)),
  Y=rep(c(0, 1, 0, 1), c(15, 10, 15, 10))
)

# 查看life数据集
View(life)

glm.sol <- glm(Y~X1+X2+X3, family = binomial, data = life)
summary(glm.sol)

# 若无巩固治疗，即X3=0，则1年以上的存活概率
pre <- predict(glm.sol, data.frame(X1=5, X2=2, X3=0))
p <- exp(pre) / (1+exp(pre))
p

# 进行巩固治疗
pre_new <- predict(glm.sol, data.frame(X1=5, X2=2, X3=1))
p_new <- exp(pre_new) / (1+exp(pre_new))
p_new

# 参数没有通过检验
# 用step()函数进行筛选
glm.new <- step(glm.sol)
summary(glm.new)

# 做预测分析
pre <- predict(glm.new, data.frame(X2=2, X3=0))
p <- exp(pre) / (1+exp(pre))
p

pre <- predict(glm.new, data.frame(X2=2, X3=1))
p <- exp(pre) / (1+exp(pre))
p

# 回归诊断
Reg_Diag<-function(fm){
  n<-nrow(fm$model); df<-fm$df.residual
  p<-n-df-1; s<-rep(" ", n);
  res<-residuals(fm); s1<-s; s1[abs(res)==max(abs(res))]<-"*"
  sta<-rstandard(fm); s2<-s; s2[abs(sta)>2]<-"*"
  stu<-rstudent(fm); s3<-s; s3[abs(sta)>2]<-"*"
  h<-hatvalues(fm); s4<-s; s4[h>2*(p+1)/n]<-"*"
  d<-dffits(fm); s5<-s; s5[abs(d)>2*sqrt((p+1)/n)]<-"*"
  c<-cooks.distance(fm); s6<-s; s6[c==max(c)]<-"*"
  co<-covratio(fm); abs_co<-abs(co-1)
  s7<-s; s7[abs_co==max(abs_co)]<-"*"
  data.frame(residual=res, s1, standard=sta, s2,
             student=stu, s3, hat_matrix=h, s4,
             DFFITS=d, s5,cooks_distance=c, s6,
             COVRATIO=co, s7)
}

Reg_Diag(glm.sol)

# 其他分部族
# poisson分部

x <- rnorm(100)
y <- rpois(100, exp(1+x))
glm(y~x, family = poisson)


# Gamma分部族

# quasi分布族
x <- rnorm(100)
y <- rpois(100, exp(1+x))
glm(y ~ x, family=quasi(var="mu", link="log"))

glm(y ~ x, family=quasi(var="mu^2", link="log"))

y <- rbinom(100, 1, plogis(x))
glm(y ~ x, family=quasi(var="mu(1-mu)", link="logit"), start=c(0,1))


# 非线性回归模型

# 多项式回归模型
alloy<-data.frame(
  x=c(37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0,
      40.5, 41.0, 41.5, 42.0, 42.5, 43.0),
  y=c(3.40, 3.00, 3.00, 3.27, 2.10, 1.83, 1.53,
      1.70, 1.80, 1.90, 2.35, 2.54, 2.90)
)

alloy

lm.sol <- lm(y~1+x+I(x^2), data=alloy)
summary(lm.sol)

xfit <- seq(37,43,len=200)
yfit <- predict(lm.sol, data.frame(x=xfit))
plot(alloy$x, alloy$y)
lines(xfit, yfit)


# 正交多项式回归
poly(alloy$x, degree = 2)

lm.pol <- lm(y~1+poly(x,2), data = alloy)
summary(lm.pol)

xfit <- seq(37,43,len=200)
yfit <- predict(lm.pol, data.frame(x=xfit))

# 内在非线性回归模型

# 非线性模型的参数估计
cl<-data.frame(
  X=c(rep(2*4:21, c(2, 4, 4, 3, 3, 2, 3, 3, 3, 3, 2,
                    3, 2, 1, 2, 2, 1, 1))),
  Y=c(0.49, 0.49, 0.48, 0.47, 0.48, 0.47, 0.46, 0.46,
      0.45, 0.43, 0.45, 0.43, 0.43, 0.44, 0.43, 0.43,
      0.46, 0.45, 0.42, 0.42, 0.43, 0.41, 0.41, 0.40,
      0.42, 0.40, 0.40, 0.41, 0.40, 0.41, 0.41, 0.40,
      0.40, 0.40, 0.38, 0.41, 0.40, 0.40, 0.41, 0.38,
      0.40, 0.40, 0.39, 0.39))
cl

nls.sol <- nls(Y~a+(0.49-a)*exp(-b*(X-8)), data = cl, start = list(a=0.1,b=0.01))
nls.summ <-summary(nls.sol)
nls.summ

xfit<-seq(8,44,len=200)
yfit<-predict(nls.sol, data.frame(X=xfit))
plot(cl$X, cl$Y); lines(xfit,yfit)


nls.summ$sigma

fn<-function(a, b, X){
  f1 <- 1-exp(-b*(X-8))
  f2 <- -(0.49-a)*(X-8)*exp(-b*(X-8))
  cbind(f1,f2)
}

D<-fn(nls.summ$parameters[1,1],
      nls.summ$parameters[2,1], cl$X)

theta.var<-nls.summ$sigma^2*solve(t(D)%*%D)
theta.var
nls.summ$parameters[,2]

# 区间估计
paramet.int<-function(fm, alpha=0.05){
  paramet <- fm$parameters[,1]
  df <- nls.summ$df[2]
  left <- paramet-nls.summ$parameters[,2]
  right <- paramet+nls.summ$parameters[,2]
  rowname <- dimnames(nls.summ$parameters)[[1]]
  colname <- c("Estimate", "Left", "Right")
  matrix(c(paramet,left, right), ncol=3,
         dimnames = list(rowname, colname ))
}

paramet.int(nls.sol)


# 函数nlm的使用



# 方差分析

# 方差分析表的计算
# 函数aov()提供了方差分析表的计算
lamp<-data.frame(
  X=c(1600, 1610, 1650, 1680, 1700, 1700, 1780, 1500, 1640,
      1400, 1700, 1750, 1640, 1550, 1600, 1620, 1640, 1600,
      1740, 1800, 1510, 1520, 1530, 1570, 1640, 1600),
  A=factor(c(rep(1,7),rep(2,5), rep(3,8), rep(4,6)))
)

View(lamp)

lamp.aov <- aov(X~A, data = lamp)
summary(lamp.aov)

# 得到完整的方差分析表
anova.tab<-function(fm){
  tab<-summary(fm)
  k<-length(tab[[1]])-2
  temp<-c(sum(tab[[1]][,1]), sum(tab[[1]][,2]), rep(NA,k))
  tab[[1]]["Total",]<-temp
  tab
}

anova.tab(lamp.aov)

plot(lamp$X~lamp$A)

# 小白鼠实验
mouse<-data.frame(
  X=c( 2, 4, 3, 2, 4, 7, 7, 2, 2, 5, 4, 5, 6, 8, 5, 10, 7,
       12, 12, 6, 6, 7, 11, 6, 6, 7, 9, 5, 5, 10, 6, 3, 10),
  A=factor(c(rep(1,11),rep(2,10), rep(3,12)))
)

mouse.aov <- aov(X~A, data = mouse)
anova.tab(mouse.aov)

# 均值的多重比较的计算
# pairwise.t.test()可以多重比较p的值

attach(mouse)
mu <- c(mean(X[A==1]), mean(X[A==2]), mean(X[A==3]))
mu

# 做多重t检验
pairwise.t.test(X, A, p.adjust.method = "none")

# p值调整方法
# holm调整方法
pairwise.t.test(X, A, p.adjust.method = "holm")

# bonferroni调整方法
pairwise.t.test(X, A, p.adjust.method = "bonferroni")


plot(mouse$X~mouse$A)


# 方差的齐次性检验
# 1 误差正态性检验
# W正态检验
# shapiro.test()

attach(lamp)

# 水平1
shapiro.test(X[A==1])

# 水平2
shapiro.test(X[A==2])

# 水平3
shapiro.test(X[A==3])

# 水平4
shapiro.test(X[A==4])


# 方差齐次性检验
# 方差齐次性检验最常用的方法是Bartlett检验
# bartlett.test()函数提供了Bartlett检验

lamp<-data.frame(
  X=c(1600, 1610, 1650, 1680, 1700, 1700, 1780, 1500, 1640,
      1400, 1700, 1750, 1640, 1550, 1600, 1620, 1640, 1600,
      1740, 1800, 1510, 1520, 1530, 1570, 1640, 1600),
  A=factor(c(rep(1,7),rep(2,5), rep(3,8), rep(4,6)))
)

# 对数据lamp做方差齐性检验
bartlett.test(X~A, data=lamp)

# Kruskal-Waills 秩和检验

food<-data.frame(
  x=c(164, 190, 203, 205, 206, 214, 228, 257,
      185, 197, 201, 231,
      187, 212, 215, 220, 248, 265, 281,
      202, 204, 207, 227, 230, 276),
  g=factor(rep(1:4, c(8,4,7,6)))
)

kruskal.test(x~g, data=food)

kruskal.test(food$x, food$g)


A<-c(164, 190, 203, 205, 206, 214, 228, 257)
B<-c(185, 197, 201, 231)
C<-c(187, 212, 215, 220, 248, 265, 281)
D<-c(202, 204, 207, 227, 230, 276)
kruskal.test(list(A,B,C,D))


# 对上述数据做正态性检验和方差齐性检验
attach(food)

# 水平1的正态性检验
shapiro.test(x[g==1])
# 水平2的正态性检验
shapiro.test(x[g==2])
# 水平3的正态性检验
shapiro.test(x[g==3])
# 水平4的正态性检验
shapiro.test(x[g==4])

# bartlett.test方差齐性检验
bartlett.test(x~g, data = food)


# Friedman 秩和检验
# friedman.test()
X<-matrix(
  c(1.00, 1.01, 1.13, 1.14, 1.70, 2.01, 2.23, 2.63,
    0.96, 1.23, 1.54, 1.96, 2.94, 3.68, 5.59, 6.96,
    2.07, 3.72, 4.50, 4.90, 6.00, 6.84, 8.23, 10.33),
  ncol=3, dimnames=list(1:8, c("A", "B", "C"))
)

friedman.test(X)

# 另外一种写法
x<-c(1.00, 1.01, 1.13, 1.14, 1.70, 2.01, 2.23, 2.63,
     0.96, 1.23, 1.54, 1.96, 2.94, 3.68, 5.59, 6.96,
     2.07, 3.72, 4.50, 4.90, 6.00, 6.84, 8.23, 10.33)
g<-gl(3,8)
b<-gl(8,1,24)
friedman.test(x,g,b)

# 再另外一种写法
mouse<-data.frame(
  x=c(1.00, 1.01, 1.13, 1.14, 1.70, 2.01, 2.23, 2.63,
      0.96, 1.23, 1.54, 1.96, 2.94, 3.68, 5.59, 6.96,
      2.07, 3.72, 4.50, 4.90, 6.00, 6.84, 8.23, 10.33),
  g=gl(3,8),
  b=gl(8,1,24)
)
friedman.test(x~g|b, data=mouse)

# 双因素分析

# 方差分析表的计算
# 用数据框的形式输入数据
agriculture<-data.frame(
  Y=c(325, 292, 316, 317, 310, 318,
      310, 320, 318, 330, 370, 365),
  A=gl(4,3),
  B=gl(3,1,12)
)

# 作双因素方差分析
agriculture.aov <- aov(Y ~ A+B, data=agriculture)

# 正交实验设计
rate<-data.frame(
  A=gl(3,3),
  B=gl(3,1,9),
  C=factor(c(1,2,3,2,3,1,3,1,2)),
  Y=c(31, 54, 38, 53, 49, 42, 57, 62, 64)
)

# 计算各个因素的均值
K <- matrix(0, nrow=3, ncol=3, dimnames = list(1:3, c("A","B","C")))
for (j in 1:3)
  for (i in 1:3)
    K[i,j] <- mean(rate$Y[rate[j]==i])
K

plot(as.vector(K), axes=F, xlab="Level", ylab="Rate")
xmark <- c(NA,"A1","A2","A3","B1","B2","B3","C1","C2","C3",NA)
axis(1,0:10,labels = xmark)
axis(2,4*10:16)
axis(3,0:10,labels = xmark)
axis(4,4*10:16)
lines(    K[,"A"])
lines(4:6,K[,"B"])
lines(7:9,K[,"C"])


# 正交试验的方差分析
rate.aov <- aov(Y~A+B+C, data=rate)
# 得到完整的方差分析表
anova.tab<-function(fm){
  tab<-summary(fm)
  k<-length(tab[[1]])-2
  temp<-c(sum(tab[[1]][,1]), sum(tab[[1]][,2]), rep(NA,k))
  tab[[1]]["Total",]<-temp
  tab
}
# 对正交实验进行方差分析
anova.tab(rate.aov)


# 有交互作用的实验
cotton<-data.frame(
  Y=c(0.30, 0.35, 0.20, 0.30, 0.15, 0.50, 0.15, 0.40),
  A=gl(2,4), B=gl(2,2,8), C=gl(2,1,8)
)
View(cotton)

cotton.aov <- aov(Y~A+B+C+A:B+A:C+B:C, data=cotton)
anova.tab<-function(fm){
  tab<-summary(fm)
  k<-length(tab[[1]])-2
  temp<-c(sum(tab[[1]][,1]), sum(tab[[1]][,2]), rep(NA,k))
  tab[[1]]["Total",]<-temp
  tab
}
anova.tab(cotton.aov)

# 抹去模型的F统计检验量小的值
cotton.new <- aov(Y~B+C+A:C, data=cotton)
anova.tab(cotton.new)

# 先写一个函数，将各个因素的交互作用算出来
ab<-function(x,y){
  n<-length(x); z<-rep(0,n)
  for (i in 1:n)
    if (x[i]==y[i]){z[i]<-1} else{z[i]<-2}
  factor(z)
}

cotton$AC <- ab(cotton$A, cotton$C)

# 再计算各个因素的均值
K <- matrix(0, nrow = 2, ncol = 4, dimnames = list(1:2, c("A", "B", "C", "AC")))

for (j in 2:5)
  for (i in 1:2)
    K[i,j-1]<-mean(cotton$Y[cotton[j]==i])

K

# 有重复实验的方差分析
mosquito<-data.frame(
  A=gl(3, 12), B=gl(3,4,36),
  C=factor(rep(c(1,2,3,2,3,1,3,1,2),rep(4,9))),
  D=factor(rep(c(1,2,3,3,1,2,2,3,1),rep(4,9))),
  Y=c( 9.41, 7.19, 10.73, 3.73, 11.91, 11.85, 11.00, 11.72,
       10.67, 10.70, 10.91, 10.18, 3.87, 3.18, 3.80, 4.85,
       4.20, 5.72, 4.58, 3.71, 4.29, 3.89, 3.88, 4.71,
       7.62, 7.01, 6.83, 7.41, 7.79, 7.38, 7.56, 6.28,
       8.09, 8.17, 8.14, 7.49)
)

mosquito.aov<-aov(Y~A+B+C+D, data=mosquito)
anova.tab(mosquito.aov)

K<-matrix(0, nrow=3, ncol=4, dimnames=list(1:3, c("A", "B", "C", "D")))

for (j in 1:4)
  for (i in 1:3)
    K[i,j]<-mean(mosquito$Y[mosquito[j]==i])

K

# 多元统计分析
# 判别分析
discriminiant.distance <- function
(TrnX1, TrnX2, TstX = NULL, var.equal = FALSE){
  if (is.null(TstX) == TRUE) TstX <- rbind(TrnX1,TrnX2)
  if (is.vector(TstX) == TRUE) TstX <- t(as.matrix(TstX))
  else if (is.matrix(TstX) != TRUE)
    TstX <- as.matrix(TstX)
  if (is.matrix(TrnX1) != TRUE) TrnX1 <- as.matrix(TrnX1)
  if (is.matrix(TrnX2) != TRUE) TrnX2 <- as.matrix(TrnX2)
  nx <- nrow(TstX)
  blong <- matrix(rep(0, nx), nrow=1, byrow=TRUE,
                  dimnames=list("blong", 1:nx))
  mu1 <- colMeans(TrnX1); mu2 <- colMeans(TrnX2)
  if (var.equal == TRUE || var.equal == T){
    S <- var(rbind(TrnX1,TrnX2))
    w <- mahalanobis(TstX, mu2, S)
    - mahalanobis(TstX, mu1, S)
  }
  else{
    S1 < -var(TrnX1); S2 <- var(TrnX2)
    w <- mahalanobis(TstX, mu2, S2)
    - mahalanobis(TstX, mu1, S1)
  }
  for (i in 1:nx){
    if (w[i] > 0)
      blong[i] <- 1
    else
      blong[i] <- 2
  }
  blong
}

classX1<-data.frame(
  x1=c(6.60, 6.60, 6.10, 6.10, 8.40, 7.2, 8.40, 7.50,
       7.50, 8.30, 7.80, 7.80),
  x2=c(39.00,39.00, 47.00, 47.00, 32.00, 6.0, 113.00, 52.00,
       52.00,113.00,172.00,172.00),
  x3=c(1.00, 1.00, 1.00, 1.00, 2.00, 1.0, 3.50, 1.00,
       3.50, 0.00, 1.00, 1.50),
  x4=c(6.00, 6.00, 6.00, 6.00, 7.50, 7.0, 6.00, 6.00,
       7.50, 7.50, 3.50, 3.00),
  x5=c(6.00, 12.00, 6.00, 12.00, 19.00, 28.0, 18.00, 12.00,
       6.00, 35.00, 14.00, 15.00),
  x6=c(0.12, 0.12, 0.08, 0.08, 0.35, 0.3, 0.15, 0.16,
       0.16, 0.12, 0.21, 0.21),
  x7=c(20.00,20.00, 12.00, 12.00, 75.00, 30.0, 75.00, 40.00,
       40.00,180.00, 45.00, 45.00)
)

classX2<-data.frame(
  x1=c(8.40, 8.40, 8.40, 6.3, 7.00, 7.00, 7.00, 8.30,
       8.30, 7.2, 7.2, 7.2, 5.50, 8.40, 8.40, 7.50,
       7.50, 8.30, 8.30, 8.30, 8.30, 7.80, 7.80),
  x2=c(32.0 ,32.00, 32.00, 11.0, 8.00, 8.00, 8.00,161.00,
       161.0, 6.0, 6.0, 6.0, 6.00,113.00,113.00, 52.00,
       52.00, 97.00, 97.00,89.00,56.00,172.00,283.00),
  x3=c(1.00, 2.00, 2.50, 4.5, 4.50, 6.00, 1.50, 1.50,
       0.50, 3.5, 1.0, 1.0, 2.50, 3.50, 3.50, 1.00,
       1.00, 0.00, 2.50, 0.00, 1.50, 1.00, 1.00),
  x4=c(5.00, 9.00, 4.00, 7.5, 4.50, 7.50, 6.00, 4.00,
       2.50, 4.0, 3.0, 6.0, 3.00, 4.50, 4.50, 6.00,
       7.50, 6.00, 6.00, 6.00, 6.00, 3.50, 4.50),
  x5=c(4.00, 10.00, 10.00, 3.0, 9.00, 4.00, 1.00, 4.00,
       1.00, 12.0, 3.0, 5.0, 7.00, 6.00, 8.00, 6.00,
       8.00, 5.00, 5.00,10.00,13.00, 6.00, 6.00),
  x6=c(0.35, 0.35, 0.35, 0.2, 0.25, 0.25, 0.25, 0.08,
       0.08, 0.30, 0.3, 0.3, 0.18, 0.15, 0.15, 0.16,
       0.16, 0.15, 0.15, 0.16, 0.25, 0.21, 0.18),
  x7=c(75.00,75.00, 75.00, 15.0,30.00, 30.00, 30.00, 70.00,
       70.00, 30.0, 30.0, 30.0,18.00, 75.00, 75.00, 40.00,
       40.00,180.00,180.00,180.00,180.00,45.00,45.00)
)

discriminiant.distance(classX1, classX2, var.equal = TRUE)
discriminiant.distance(classX1, classX2)

# 多分类问题的距离判别
distinguish.distance <- function(TrnX, TrnG, TstX = NULL, var.equal = FALSE){
  if ( is.factor(TrnG) == FALSE){
    mx <- nrow(TrnX); mg <- nrow(TrnG)
    TrnX <- rbind(TrnX, TrnG)
    TrnG <- factor(rep(1:2, c(mx, mg)))
  }
  if (is.null(TstX) == TRUE) TstX <- TrnX
  if (is.vector(TstX) == TRUE) TstX <- t(as.matrix(TstX))
  else if (is.matrix(TstX) != TRUE)
    TstX <- as.matrix(TstX)
  if (is.matrix(TrnX) != TRUE) TrnX <- as.matrix(TrnX)
  nx <- nrow(TstX)
  blong <- matrix(rep(0, nx), nrow=1,
                  dimnames=list("blong", 1:nx))
  g <- length(levels(TrnG))
  mu <- matrix(0, nrow=g, ncol=ncol(TrnX))
  for (i in 1:g)
    mu[i,] <- colMeans(TrnX[TrnG==i,])
  D < -matrix(0, nrow=g, ncol=nx)
  if (var.equal == TRUE || var.equal == T){
    for (i in 1:g)
      D[i,] <- mahalanobis(TstX, mu[i,], var(TrnX))
  }
  else{
    for (i in 1:g)
      D[i,] <- mahalanobis(TstX, mu[i,], var(TrnX[TrnG==i,]))
  }
  for (j in 1:nx){
    dmin <- Inf
    for (i in 1:g)
      if (D[i,j] < dmin){
        dmin <- D[i,j]; blong[j] <- i
      }
  }
  blong
}

# 前四列是属性
X <- iris[, 1:4]
G <- gl(3:50)


# 贝叶斯判别
discriminiant.bayes <- function
(TrnX1, TrnX2, rate = 1, TstX = NULL, var.equal = FALSE){
  if (is.null(TstX) == TRUE) TstX<-rbind(TrnX1,TrnX2)
  if (is.vector(TstX) == TRUE) TstX <- t(as.matrix(TstX))
  else if (is.matrix(TstX) != TRUE)
    TstX <- as.matrix(TstX)
  if (is.matrix(TrnX1) != TRUE) TrnX1 <- as.matrix(TrnX1)
  if (is.matrix(TrnX2) != TRUE) TrnX2 <- as.matrix(TrnX2)
  nx <- nrow(TstX)
  blong <- matrix(rep(0, nx), nrow=1, byrow=TRUE,
                  dimnames=list("blong", 1:nx))
  mu1 <- colMeans(TrnX1); mu2 <- colMeans(TrnX2)
  if (var.equal == TRUE || var.equal == T){
    S <- var(rbind(TrnX1,TrnX2)); beta <- 2*log(rate)
    w <- mahalanobis(TstX, mu2, S)
    - mahalanobis(TstX, mu1, S)
  }
  else{
    S1 <- var(TrnX1); S2 <- var(TrnX2)
    beta <- 2*log(rate) + log(det(S1)/det(S2))
    w <- mahalanobis(TstX, mu2, S2)
    - mahalanobis(TstX, mu1, S1)
  }
  for (i in 1:nx){
    if (w[i] > beta)
      blong[i] <- 1
    else
      blong[i] <- 2
  }
  blong
}

TrnX1<-matrix(c(24.8, 24.1, 26.6, 23.5, 25.5, 27.4,-2.0, -2.4, -3.0, -1.9, -2.1, -3.1), ncol=2)
TrnX2<-matrix(
  c(22.1, 21.6, 22.0, 22.8, 22.7, 21.5, 22.1, 21.4,
    -0.7, -1.4, -0.8, -1.6, -1.5, -1.0, -1.2, -1.3),
  ncol=2)
discriminiant.bayes(X1, X2, rate=8/6, var.equal=TRUE)
discriminiant.bayes(TrnX1, TrnX2, rate=8/6)

# 聚类分析

# 生成数据结构
x<-c(1,2,6,8,11); dim(x)<-c(5,1); d<-dist(x)
# 生成系统聚类
hc1<-hclust(d, "single"); hc2<-hclust(d, "complete")
hc3<-hclust(d, "median"); hc4<-hclust(d, "mcquitty")
# 画出树状图
opar <- par(mfrow = c(2, 2))
plot(hc1,hang=-1); plot(hc2,hang=-1)
plot(hc3,hang=-1); plot(hc4,hang=-1)
par(opar)

dend1<-as.dendrogram(hc1)
opar <- par(mfrow = c(2, 2),mar = c(4,3,1,2))
plot(dend1)
plot(dend1, nodePar=list(pch = c(1,NA), cex=0.8, lab.cex=0.8),
     type = "t", center=TRUE)
plot(dend1, edgePar=list(col = 1:2, lty = 2:3),
     dLeaf=1, edge.root = TRUE)
plot(dend1, nodePar=list(pch = 2:1, cex=.4*2:1, col=2:3),
     horiz=TRUE)
par(opar)

# 输入相关矩阵
x<-c(1.000, 0.846, 0.805, 0.859, 0.473, 0.398, 0.301, 0.382,
     0.846, 1.000, 0.881, 0.826, 0.376, 0.326, 0.277, 0.277,
     0.805, 0.881, 1.000, 0.801, 0.380, 0.319, 0.237, 0.345,
     0.859, 0.826, 0.801, 1.000, 0.436, 0.329, 0.327, 0.365,
     0.473, 0.376, 0.380, 0.436, 1.000, 0.762, 0.730, 0.629,
     0.398, 0.326, 0.319, 0.329, 0.762, 1.000, 0.583, 0.577,
     0.301, 0.277, 0.237, 0.327, 0.730, 0.583, 1.000, 0.539,
     0.382, 0.415, 0.345, 0.365, 0.629, 0.577, 0.539, 1.000)
names<-c("身高", "手臂长", "上肢长", "下肢长", "体重", "颈围", "胸围", "胸宽")
r <- matrix(x, nrow = 8, dimnames = list(names, names))

# 作系统性聚类分析
d<-as.dist(1-r); hc<-hclust(d); dend<-as.dendrogram(hc)
# 使得系谱图更加好看
nP<-list(col=3:2, cex=c(2.0, 0.75), pch= 21:22,
         bg= c("light blue", "pink"),
         lab.cex = 1.0, lab.col = "tomato")
addE <- function(n){
  if(!is.leaf(n)){
    attr(n,"edgePar")<-list(p.col="plum")
    attr(n,"edgetext")<-paste(attr(n,"members"),"members")
  }
  n
}

# 画出系谱图
de <- dendrapply(dend, addE)
plot(de, nodePar=np)

# 类个数的确定
# 函数rect.hclust()


X<-data.frame(
  x1=c(2959.19, 2459.77, 1495.63, 1046.33, 1303.97, 1730.84,
       1561.86, 1410.11, 3712.31, 2207.58, 2629.16, 1844.78,
       2709.46, 1563.78, 1675.75, 1427.65, 1783.43, 1942.23,
       3055.17, 2033.87, 2057.86, 2303.29, 1974.28, 1673.82,
       2194.25, 2646.61, 1472.95, 1525.57, 1654.69, 1375.46,
       1608.82),
  x2=c(730.79, 495.47, 515.90, 477.77, 524.29, 553.90, 492.42,
       510.71, 550.74, 449.37, 557.32, 430.29, 428.11, 303.65,
       613.32, 431.79, 511.88, 512.27, 353.23, 300.82, 186.44,
       589.99, 507.76, 437.75, 537.01, 839.70, 390.89, 472.98,
       437.77, 480.99, 536.05),
  x3=c(749.41, 697.33, 362.37, 290.15, 254.83, 246.91, 200.49,
       211.88, 893.37, 572.40, 689.73, 271.28, 334.12, 233.81,
       550.71, 288.55, 282.84, 401.39, 564.56, 338.65, 202.72,
       516.21, 344.79, 461.61, 369.07, 204.44, 447.95, 328.90,
       258.78, 273.84, 432.46),
  x4=c(513.34, 302.87, 285.32, 208.57, 192.17, 279.81, 218.36,
       277.11, 346.93, 211.92, 435.69, 126.33, 160.77, 107.90,
       219.79, 208.14, 201.01, 206.06, 356.27, 157.78, 171.79,
       236.55, 203.21, 153.32, 249.54, 209.11, 259.51, 219.86,
       303.00, 317.32, 235.82),
  x5=c(467.87, 284.19, 272.95, 201.50, 249.81, 239.18, 220.69,
       224.65, 527.00, 302.09, 514.66, 250.56, 405.14, 209.70,
       272.59, 217.00, 237.60, 321.29, 811.88, 329.06, 329.65,
       403.92, 240.24, 254.66, 290.84, 379.30, 230.61, 206.65,
       244.93, 251.08, 250.28),
  x6=c(1141.82, 735.97, 540.58, 414.72, 463.09, 445.20, 459.62,
       376.82, 1034.98, 585.23, 795.87, 513.18, 461.67, 393.99,
       599.43, 337.76, 617.74, 697.22, 873.06, 621.74, 477.17,
       730.05, 575.10, 445.59, 561.91, 371.04, 490.90, 449.69,
       479.53, 424.75, 541.30),
  x7=c(478.42, 570.84, 364.91, 281.84, 287.87, 330.24, 360.48,
       317.61, 720.33, 429.77, 575.76, 314.00, 535.13, 509.39,
       371.62, 421.31, 523.52, 492.60, 1082.82, 587.02, 312.93,
       438.41, 430.36, 346.11, 407.70, 269.59, 469.10, 249.66,
       288.56, 228.73, 344.85),
  x8=c(457.64, 305.08, 188.63, 212.10, 192.96, 163.86, 147.76,
       152.85, 462.03, 252.54, 323.36, 151.39, 232.29, 160.12,
       211.84, 165.32, 182.52, 226.45, 420.81, 218.27, 279.19,
       225.80, 223.46, 191.48, 330.95, 389.33, 191.34, 228.19,
       236.51, 195.93, 214.40),
  row.names=c("北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "上海",
              "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东",
              "广西", "海南", "重庆", "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海",
              "宁夏", "新疆"
              )
  )

# x1 食品
# x2 衣着
# x3 家庭设备用品及服务
# x4 医疗保健
# x5 交通与通讯
# x6 娱乐教育文化服务
# x7 居住
# x8 杂项商品和服务

# 进行标准化
XSTD <- scale(X)

# 生成距离结构
# 做系统聚类
d <- dist(XSTD)
hc1 <- hclust(d)
hc2 <- hclust(d, "average")
hc3 <- hclust(d, "centroid")
hc4 <- hclust(d, "ward")

# 绘出系谱图和聚类情况
opar <- par(mfrow=c(2,1), mar=c(5.2, 4, 0, 0))
plclust(hc1, hang = -1); re1 <- rect.hclust(hc1, k=5, border = "red")
plclust(hc2, hang = -1); re2 <- rect.hclust(hc2, k=5, border = "red")
par(opar)

# 绘制出系谱图和聚类情况（重心法和Wald法）
opar <- par(mfrow=c(2,1), mar=c(5.2, 4, 0, 0))
plclust(hc3,hang=-1); re3<-rect.hclust(hc3,k=5,border="red")
plclust(hc4,hang=-1); re4<-rect.hclust(hc4,k=5,border="red")
par(opar)

# 动态聚类法
km <- kmeans(scale(X), 5, nstart = 20)
km

# 主成分分析
# princomp() 函数
# summary() 函数
# loadings() 函数 显示 主成分分析或者因子分析中的loadings因子载荷的内容
# predict() 函数 预测主成分的值
# screeplot() 函数 绘制出主成分的碎石图
# biplot() 函数 画出数据关于主成分的散点图和原坐标在主成分下的方向

# 用数据框的形式输入数据
student<-data.frame(
  X1=c(148, 139, 160, 149, 159, 142, 153, 150, 151, 139,
       140, 161, 158, 140, 137, 152, 149, 145, 160, 156,
       151, 147, 157, 147, 157, 151, 144, 141, 139, 148),
  X2=c(41, 34, 49, 36, 45, 31, 43, 43, 42, 31,
       29, 47, 49, 33, 31, 35, 47, 35, 47, 44,
       42, 38, 39, 30, 48, 36, 36, 30, 32, 38),
  X3=c(72, 71, 77, 67, 80, 66, 76, 77, 77, 68,
       64, 78, 78, 67, 66, 73, 82, 70, 74, 78,
       73, 73, 68, 65, 80, 74, 68, 67, 68, 70),
  X4=c(78, 76, 86, 79, 86, 76, 83, 79, 80, 74,
       74, 84, 83, 77, 73, 79, 79, 77, 87, 85,
       82, 78, 80, 75, 88, 80, 76, 76, 73, 78)
)

# 做主成分分析，并且显示分析结果
student.pr <- princomp(student, cor = TRUE)
summary(student.pr, loadings=TRUE)

# 做预测
predict(student.pr)

# 画出碎石图
screeplot(student.pr, type = "lines")
screeplot(student.pr, type = "barplot")

#  主成分分析的应用
# 输入相关性矩阵

x<-c(1.00,
     0.79, 1.00,
     0.36, 0.31, 1.00,
     0.96, 0.74, 0.38, 1.00,
     0.89, 0.58, 0.31, 0.90, 1.00,
     0.79, 0.58, 0.30, 0.78, 0.79, 1.00,
     0.76, 0.55, 0.35, 0.75, 0.74, 0.73, 1.00,
     0.26, 0.19, 0.58, 0.25, 0.25, 0.18, 0.24, 1.00,
     0.21, 0.07, 0.28, 0.20, 0.18, 0.18, 0.29,-0.04, 1.00,
     0.26, 0.16, 0.33, 0.22, 0.23, 0.23, 0.25, 0.49,-0.34, 1.00,
     0.07, 0.21, 0.38, 0.08,-0.02, 0.00, 0.10, 0.44,-0.16, 0.23, 1.00,
     0.52, 0.41, 0.35, 0.53, 0.48, 0.38, 0.44, 0.30,-0.05, 0.50, 0.24, 1.00,
     0.77, 0.47, 0.41, 0.79, 0.79, 0.69, 0.67, 0.32, 0.23, 0.31, 0.10, 0.62, 1.00,
     0.25, 0.17, 0.64, 0.27, 0.27, 0.14, 0.16, 0.51, 0.21, 0.15, 0.31, 0.17, 0.26, 1.00,
     0.51, 0.35, 0.58, 0.57, 0.51, 0.26, 0.38, 0.51, 0.15, 0.29, 0.28, 0.41, 0.50, 0.63, 1.00,
     0.21, 0.16, 0.51, 0.26, 0.23, 0.00, 0.12, 0.38, 0.18, 0.14, 0.31, 0.18, 0.24, 0.50, 0.65, 1.00)

# 输入变量名称
names <- c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16")

# 将矩阵生成相关矩阵
R <- matrix(0, nrow=16, ncol = 16, dimnames = list(names, names))
R
# 将向量生成相关性矩阵
for (i in 1:16)
{
  for (j in 1:16)
  {
    R[i,j] <- x[(i-1)*i/2+j]
    R[j,i] <- R[i,j]
  }
}

R
View(R)

# 做主成分分析
pr <- princomp(covmat=R)
load <- loadings(pr)

# 画散点图
plot(load[,1:2])
text(load[,1], load[,2], adj=c(-0.4, 0.3))

# 主成分回归
# 法国经济数据
# 用数据框的形式输入数据
conomy<-data.frame(
  x1=c(149.3, 161.2, 171.5, 175.5, 180.8, 190.7,
       202.1, 212.4, 226.1, 231.9, 239.0),
  x2=c(4.2, 4.1, 3.1, 3.1, 1.1, 2.2, 2.1, 5.6, 5.0, 5.1, 0.7),
  x3=c(108.1, 114.8, 123.2, 126.9, 132.1, 137.7,
       146.0, 154.1, 162.3, 164.3, 167.6),
  y=c(15.9, 16.4, 19.0, 19.1, 18.8, 20.4, 22.7,
      26.5, 28.1, 27.6, 26.3)
)

# 做线性回归
lm.sol <- lm(y~x1+x2+x3, data=conomy)
summary(lm.sol)

# 为了克服多重共线性的影响，对变量做主成分分析
conomy.pr <- princomp(~x1+x2+x3, data=conomy, cor=T)
summary(conomy.pr, loadings = TRUE)

# 预测主成分，并且做回归分析
pre <- predict(conomy.pr)
conomy$z1 <- pre[,1]
conomy$z2 <- pre[,2]
lm.sol <- lm(y~z1+z2, data=conomy)
summary(lm.sol)

# 做变换，得到原坐标下的关系表达式
beta <- coef(lm.sol)
A <- loadings(conomy.pr)
x.bar <- conomy.pr$center
x.sd <- conomy.pr$scale
coef <- (beta[2]*A[,1] + beta[3]*A[,2]) / x.sd
beta0 <- beta[1] - sum(x.bar * coef)
# 得到相应的系数
c(beta0, coef)

# 因子分析
# 主成分法
factor.analy1<-function(S, m){
  p<-nrow(S); diag_S<-diag(S); sum_rank<-sum(diag_S)
  rowname<-paste("X", 1:p, sep="")
  colname<-paste("Factor", 1:m, sep="")
  A<-matrix(0, nrow=p, ncol=m,
            dimnames=list(rowname, colname))
  eig<-eigen(S)
  for (i in 1:m)
    A[,i]<-sqrt(eig$values[i])*eig$vectors[,i]
  h<-diag(A%*%t(A))
  rowname<-c("SS loadings","Proportion Var","Cumulative Var")
  B<-matrix(0, nrow=3, ncol=m,
            dimnames=list(rowname, colname))
  for (i in 1:m){
    B[1,i]<-sum(A[,i]^2)
    B[2,i]<-B[1,i]/sum_rank
    B[3,i]<-sum(B[1,1:i])/sum_rank
  }
  method<-c("Principal Component Method")
  list(method=method, loadings=A,
       var=cbind(common=h, spcific=diag_S-h), B=B)
}

# 输入相关矩阵
x<-c(1.000,
     0.923, 1.000,
     0.841, 0.851, 1.000,
     0.756, 0.807, 0.870, 1.000,
     0.700, 0.775, 0.835, 0.918, 1.000,
     0.619, 0.695, 0.779, 0.864, 0.928, 1.000,
     0.633, 0.697, 0.787, 0.869, 0.935, 0.975, 1.000,
     0.520, 0.596, 0.705, 0.806, 0.866, 0.932, 0.943, 1.000)

names <- c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8")
R <- matrix(0, nrow=8, ncol=8, dimnames = list(names, names))
R
for (i in 1:8)
{
  for (j in 1:8)
  {
    R[i,j] <- x[(i-1)*i/2+j]
    R[j,i] <- R[i,j]
  }
}
R

fa <- factor.analy1(R, m=2)
fa

E<- R-fa$loadings %*% t(fa$loadings)-diag(fa$var[,2])
sum(E^2)

# 主因子法
factor.analy2<-function(R, m, d){
  p<-nrow(R); diag_R<-diag(R); sum_rank<-sum(diag_R)
  rowname<-paste("X", 1:p, sep="")
  colname<-paste("Factor", 1:m, sep="")
  A<-matrix(0, nrow=p, ncol=m,
            dimnames=list(rowname, colname))
  kmax=20; k<-1; h <- diag_R-d
  repeat{
    diag(R)<- h; h1<-h; eig<-eigen(R)
    for (i in 1:m)
      A[,i]<-sqrt(eig$values[i])*eig$vectors[,i]
    h<-diag(A %*% t(A))
    if ((sqrt(sum((h-h1)^2))<1e-4)|k==kmax) break
    k<-k+1
  }
  rowname<-c("SS loadings","Proportion Var","Cumulative Var")
  B<-matrix(0, nrow=3, ncol=m,
            dimnames=list(rowname, colname))
  for (i in 1:m){
    B[1,i]<-sum(A[,i]^2)
    B[2,i]<-B[1,i]/sum_rank
    B[3,i]<-sum(B[1,1:i])/sum_rank
  }
  method<-c("Principal Factor Method")
  list(method=method, loadings=A,
       var=cbind(common=h,spcific=diag_R-h),B=B,iterative=k)
}

d<-c(0.123, 0.112, 0.155, 0.116, 0.073, 0.045, 0.033, 0.095)
fa <- factor.analy2(R, m=2, d)
fa

E<- R-fa$loadings %*% t(fa$loadings)-diag(fa$var[,2])
sum(E^2)

# 极大似然法
factor.analy3<-function(S, m, d){
  p<-nrow(S); diag_S<-diag(S); sum_rank<-sum(diag_S)
  rowname<-paste("X", 1:p, sep="")
  colname<-paste("Factor", 1:m, sep="")
  A<-matrix(0, nrow=p, ncol=m,
            dimnames=list(rowname, colname))
  kmax=20; k<-1
  repeat{
    d1<-d; d2<-1/sqrt(d); eig<-eigen(S * (d2 %o% d2))
    for (i in 1:m)
      A[,i]<-sqrt(eig$values[i]-1)*eig$vectors[,i]
    A<-diag(sqrt(d)) %*% A
    d<-diag(S-A%*%t(A))
    if ((sqrt(sum((d-d1)^2))<1e-4)|k==kmax) break
    k<-k+1
  }
  rowname<-c("SS loadings","Proportion Var","Cumulative Var")
  B<-matrix(0, nrow=3, ncol=m,
            dimnames=list(rowname, colname))
  for (i in 1:m){
    B[1,i]<-sum(A[,i]^2)
    B[2,i]<-B[1,i]/sum_rank
    B[3,i]<-sum(B[1,1:i])/sum_rank
  }
  method<-c("Maximum Likelihood Method")
  list(method=method, loadings=A,
       var=cbind(common=diag_S-d, spcific=d),B=B,iterative=k)
}

d<-c(0.123, 0.112, 0.155, 0.116, 0.073, 0.045, 0.033, 0.095)
fa<-factor.analy3(R, m=2, d); fa

# 主成分法、主因子法、最大似然法
factor.analy<-function(S, m=0,
                       d=1/diag(solve(S)), method="likelihood"){
  if (m==0){
    p<-nrow(X); eig<-eigen(S)
    sum_eig<-sum(diag(S))
    for (i in 1:p){
      if (sum(eig$values[1:i])/sum_eig>0.70){
        m<-i; break
      }
    }
  }
  
  switch(method,
         princomp=factor.analy1(S, m),
         factor=factor.analy2(S, m, d),
         likelihood=factor.analy3(S, m, d)
  )
}

# 主因子法
fa<-factor.analy(R, m=2, method="princomp")
vm1<-varimax(fa$loadings, normalize = F); vm1

# 主成分法
fa<-factor.analy(R, m=2, method="factor")
vm1<-varimax(fa$loadings, normalize = F); vm1

# 最大似然法
fa <- factor.analy(R, m=2, method = "likehood")
vm1 <- varimax(fa$loadings, normalize = F)

# 因子分析的计算函数
# R语言中提供了因子分析的计算函数
# 函数factanal()， 它可以从样本数据，样本方差矩阵和相关矩阵出发对数据作因子分析
# 并且直接给出方差最大的载荷因子矩阵






# 典型相关分析
# 典型相关分析函数
# cancor()

# 输入数据
test<-data.frame(
  X1=c(191, 193, 189, 211, 176, 169, 154, 193, 176, 156,
       189, 162, 182, 167, 154, 166, 247, 202, 157, 138),
  X2=c(36, 38, 35, 38, 31, 34, 34, 36, 37, 33,
       37, 35, 36, 34, 33, 33, 46, 37, 32, 33),
  X3=c(50, 58, 46, 56, 74, 50, 64, 46, 54, 54,
       52, 62, 56, 60, 56, 52, 50, 62, 52, 68),
  Y1=c( 5, 12, 13, 8, 15, 17, 14, 6, 4, 15,
        2, 12, 4, 6, 17, 13, 1, 12, 11, 2),
  Y2=c(162, 101, 155, 101, 200, 120, 215, 70, 60, 225,
       110, 105, 101, 125, 251, 210, 50, 210, 230, 110),
  Y3=c(60, 101, 58, 38, 40, 38, 105, 31, 25, 73,
       60, 37, 42, 40, 250, 115, 50, 120, 80, 43)
)

test <- scale(test) # 数据标准化
ca <- cancor(test[ ,1:3], test[ ,4:6])
ca

U <- as.matrix(test[,1:3]) %*% ca$xcoef
V <- as.matrix(test[,4:6]) %*% ca$ycoef

# 画出相关变量U1, V1和U3, V3为坐标的数据散点图
plot(U[,1], V[,1], xlab = "U1", ylab = "V1")
plot(U[,3], V[,3], xlab = "V3", ylab = "V3")

# 典型相关系数的显著性检验
# 相关系数检验的R程序
corcoef.test<-function(r, n, p, q, alpha=0.1){
  m<-length(r); Q<-rep(0, m); lambda <- 1
  for (k in m:1){
    lambda<-lambda*(1-r[k]^2);
    Q[k]<- -log(lambda)
  }
  s<-0; i<-m
  for (k in 1:m){
    Q[k]<- (n-k+1-1/2*(p+q+3)+s)*Q[k]
    chi<-1-pchisq(Q[k], (p-k+1)*(q-k+1))
    if (chi>alpha){
      i<-k-1; break
    }
    s<-s+1/r[k]^2
  }
  i
}

corcoef.test(r=ca$cor, n=20, p=3, q=3)

# 计算机模拟方法
# 概率分析
# 蒙特卡罗方法
# Buffon掷针问题
buffon<-function(n, l=0.8, a=1){
  k<-0
  theta<-runif(n, 0, pi); x<-runif(n, 0, a/2)
  for (i in 1:n){
    if (x[i]<= l/2*sin(theta[i]))
      k<-k+1
  }
  2*l*n/(k*a)
}

# 调用已经编好的，进行模拟
buffon(100000,l=0.8,a=1)

# 蒙特卡罗求定积分
MC1 <- function(n){
  k <- 0; x <- runif(n); y <- runif(n)
  for (i in 1:n){
    if (x[i]^2+y[i]^2 < 1)
      k <- k+1
  }
  4*k/n
}

MC1(100000)

MC1_2 <- function(n){
  x <- runif(n)
  4*sum(sqrt(1-x^2))/n
}

MC1_2(100000)


# 系统模拟
plot(c(0,1,1,0), c(0,0,1,1), xlab =" ", ylab = " ")
text(0, 1, labels="A", adj=c( 0.3, 1.3))
text(1, 1, labels="B", adj=c( 1.5, 0.5))
text(1, 0, labels="C", adj=c( 0.3, -0.8))
text(0, 0, labels="D", adj=c(-0.5, 0.1))
points(0.5,0.5); text(0.5,0.5,labels="O",adj=c(-1.0,0.3))

delta_t<-0.01; n=110
x<-matrix(0, nrow=5, ncol=n); x[,1]<-c(0,1,1,0,0)
y<-matrix(0, nrow=5, ncol=n); y[,1]<-c(1,1,0,0,1)
d<-c(0,0,0,0)
for (j in 1:(n-1)){
  for (i in 1:4){
    d[i]<-sqrt((x[i+1, j]-x[i,j])^2+(y[i+1, j]-y[i,j])^2)
    x[i,j+1]<-x[i,j]+delta_t*(x[i+1,j]-x[i,j])/d[i]
    y[i,j+1]<-y[i,j]+delta_t*(y[i+1,j]-y[i,j])/d[i]
  }
  x[5,j+1]<-x[1, j+1]; y[5, j+1]<-y[1, j+1]
}

for (i in 1:4) lines(x[i,], y[i,])













