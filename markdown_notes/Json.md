# Json

*轻量级文本数据交换格式*

*独立语言*

##### what r this?

==J==ava ==S==cript ==O==bject ==N==otation

```json
{"firstName"":"John" , "lastName":"Doe}
```

key and value都有意义

```
{
	"enployees":[
		{"firstName"":"John" , "lastName":"Doe"},
		{"firstName"":"Anna" , "lastName":"Doe"},
		{"firstName"":"Elsa" , "lastName":"Doe"}
	]
}
```

##### py中js的作用

1. 使用js字符串生成py对象（load）  <!--生成字典或者列表-->
2. 由py对象格式化成为js字符串（dump）

##### 数据类型转换

| py         | js     |
| ---------- | ------ |
| dict       | object |
| list,tuple | array  |
| str        | string |
| int, float | number |
| True       | true   |
| False      | false  |
| None       | null   |

<!--长得有点像C++ ？-->

##### 使用方法

| 方法              | 功能                                 |
| ----------------- | ------------------------------------ |
| json.dump(obj,fp) | 将py数据类型转换并保存到js格式文件内 |
| json.dumps(obj)   | 将py数据转换为js字符串               |
| json.load(fp)     | 从js文件中读取数据并转换为py类型     |
| json.loads(s)     | js字符串转换为py的类型               |

##### 针对给到的data，在py中应该何使用

```python
with open(json_file,'r', encoding = 'utf-8') as f:  # 打开文件，权限为只读，自动关闭
	data = json.load(f)  # 将js数据加载到py中
	
for group in data:  # 每个数据组读取数据
	vectors = group['vectors']
	ori_axis = group['ori_axis']
	tasks = group['tasks']
	
	cs = CoordinateSystem(vectors, ori_axis)
	
	# 依次执行任务
	for idx, tasd in enumerate(task, 1):
		task_type = task['type']
		
		if task_type == 'change_axis':  # 逐个判断任务类型
			new_axis = task['obj_axis']  # 调用封装好的类里面的方法转换坐标系
			cs.change_axis(new_axis)
		elif task_type == 'axis_angle':
			ang = cs.angle()  # 有与维度数相等的夹角数
			for i, an in enuemrate(ang):
				print(f"向量{i + 1}: {an}")
				
		elif task_type == 'area':
			area_val = cs.area()
		else:
			print(f"未知任务类型{task_type}")
			
			
```

