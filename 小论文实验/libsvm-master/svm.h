#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 324

#ifdef __cplusplus
extern "C" {
#endif

extern int libsvm_version;

//存储单一向量中的单个特征
struct svm_node
{
	int index;		//特征索引
	double value;	//特征的值
};

//存储整个样本
struct svm_problem
{
	int l;		//记录样本总数
	double *y;	//样本所属标签类别的数组
	struct svm_node **x;	//存储样本内容除标签外的信息
};

/*
SVM的种类
	- C_SVC：C表示惩罚因子，C越大表示对错误分类的惩罚越大
	- NU_SVC：和C_SVC相同
	- ONE_CLASS:不需要类标号,用于支持向量的密度估计和聚类
	- EPSILON_SVR:-不敏感损失函数，对样本点来说，存在着一个不为目标函数提供任何损失值的区域，即-带
	- NU_SVR：由于EPSILON_SVR需要事先确定参数，然而在某些情况下选择合适的参数却不是一件容易的事情。而NU_SVR能够自动计算参数
*/
enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };

/*
核函数的种类
	- LINEAR：线性核函数（linear kernel）
	- POLY:多项式核函数（ploynomial kernel）
	- RBF:径向机核函数(radical basis function)
	- SIGMOID: 神经元的非线性作用函数核函数(Sigmoid tanh)
	- PRECOMPUTED：用户自定义核函数
*/
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED };

struct svm_parameter
{
	int svm_type; /* SVM分类器的种类 */
	int kernel_type;
	int degree;	/* for poly，多项式核 */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */

	/* these are for training only */
	double cache_size; /* in MB */
	double eps;	/* stopping criteria */
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	double* weight;		/* for C_SVC */
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */
	int probability; /* do probability estimates */
};

//
// svm_model
//
struct svm_model
{
	struct svm_parameter param;	/* parameter */
	int nr_class;		/* number of classes, = 2 in regression/one class svm */
	int l;			/* total #SV */
	struct svm_node **SV;		/* SVs (SV[l]) */
	double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
	double *probA;		/* pariwise probability information */
	double *probB;
	int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

	/* for classification only */

	int *label;		/* label of each class (label[k]) */
	int *nSV;		/* number of SVs for each class (nSV[k]) */
				/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
	/* XXX */
	int free_sv;		/* 1 if svm_model is created by svm_load_model*/
				/* 0 if svm_model is created by svm_train */
};

struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

int svm_save_model(const char *model_file_name, const struct svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);
int svm_get_nr_sv(const struct svm_model *model);
double svm_get_svr_probability(const struct svm_model *model);

double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
double svm_predict(const struct svm_model *model, const struct svm_node *x);
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

void svm_free_model_content(struct svm_model *model_ptr);
void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
void svm_destroy_param(struct svm_parameter *param);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
int svm_check_probability_model(const struct svm_model *model);

void svm_set_print_string_function(void (*print_func)(const char *));

#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */
