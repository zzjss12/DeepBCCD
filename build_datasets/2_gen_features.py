#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys

import networkx as nx
import config
import config_for_feature
import idaapi
import idautils
import idc
import csv
import os
import json
import time
import shutil
from miasm2.core.bin_stream_ida import bin_stream_ida
from miasm2.core.asmblock import expr_is_label, AsmLabel, is_int
from miasm2.expression.simplifications import expr_simp
from miasm2.analysis.data_flow import dead_simp
from miasm2.ir.ir import AssignBlock, IRBlock
from utils import guess_machine, expr2colorstr
import re
from _ctypes import PyObj_FromPtr

from DL.pre_process import normalization,bracket_splict
idaapi.autoWait()

bin_num = 0
func_num = 0
function_list_file = ""
function_list_fp = None
functions=[]#由于windows文件名不区分大小写，这里记录已经分析的函数名（全部转换成小写，若重复，则添加当前时间戳作为后缀）

curBinNum = 0


class bbls:
	id=""
	define=[]
	use=[]
	defuse={}
	fathernode=set()
	childnode=set()
	define=set()
	use=set()
	visited=False
youhua=['O0','O1','O2','O3','Os']
register_Total_list = [ 'rax', 'rcx', 'rdx', 'rbx', 'rsi', 'rdi', 'rsp', 'rbp', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
                        'eax', 'ecx', 'edx', 'ebx', 'esi', 'edi', 'esp', 'ebp', 'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d',
                        'ax', 'cx', 'dx', 'bx', 'si', 'di', 'sp', 'bp', 'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w',
                        'al', 'cl', 'dl', 'bl', 'sil', 'dil', 'spl', 'bpl', 'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b']

register_list_8_byte = ['rax', 'rcx', 'rdx', 'rbx', 'rsi', 'rdi', 'rsp', 'rbp', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']
register_list_4_byte = ['eax', 'ecx', 'edx', 'ebx', 'esi', 'edi', 'esp', 'ebp', 'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d']

register_list_2_byte = ['ax', 'cx', 'dx', 'bx', 'si', 'di', 'sp', 'bp', 'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w']

register_list_1_byte = ['al', 'cl', 'dl', 'bl', 'sil', 'dil', 'spl', 'bpl', 'r8b', 'r9b', 'r10b',
                        'r11b', 'r12b', 'r13b', 'r14b', 'r15b']

###cs//ds//es//ss//fs//gs//段寄存器----------


__reg_name_list = ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp', 'eip']
__call_ins_list = ['call', 'callfi', 'callni']
__jump_ins_list = ['ja', 'jae', 'jb', 'jbe', 'jc', 'jcxz', 'jecxz', 'jrcxz', 'je', 'jg', 'jge', 'jl',
                        'jle', 'jna', 'jnae', 'jnb', 'jnbe', 'jnc', 'jne', 'jng', 'jnge', 'jnl', 'jnle', 'jno',
                        'jnp', 'jns', 'jnz', 'jo', 'jp', 'jpe', 'jpo', 'js', 'jz', 'jmp', 'jmpfi', 'jmpni', 'jmpshort']
__load_ins_list = ['lea', 'leavew', 'leave', 'leaved', 'leaveq', 'lds', 'les', 'lfs', 'lgs', 'lss']
__other_ins_list = ['nop', 'popf', 'pushf', 'retn', 'cdq', 'fldz', 'leave']

def calConstantNumber(ea):
	i = 0
	curStrNum = 0
	numeric = 0
	#print() idc.GetDisasm(ea)
	while i <= 1:
		if (idc.GetOpType(ea,i ) == 5):
			addr = idc.GetOperandValue(ea, i)
			if (idc.SegName(addr) == '.rodata') and (idc.GetType(addr) == 'char[]') and (i == 1):
				curStrNum = curStrNum + 1
			else :
				numeric = numeric + 1
		i = i + 1
	return numeric,curStrNum
def calBasicBlockFeature_gemini(block):
	numericNum = 0
	stringNum = 0
	transferNum = 0
	callNum = 0
	InstrNum = 0
	arithNum = 0
	logicNum = 0
	curEA = block.startEA
	while curEA <= block.endEA :
		#	数值常量 , 字符常量的数量
		numer, stri = calConstantNumber(curEA)
		numericNum = numericNum + numer
		stringNum = stringNum + stri
		#	转移指令的数量
		if idc.GetMnem(curEA) in config_for_feature.Gemini_allTransferInstr:
			transferNum = transferNum + 1
		# 调用的数量
		if idc.GetMnem(curEA) == 'call':
			callNum = callNum + 1
		# 指令的数量
		InstrNum = InstrNum + 1
		#	算术指令的数量
		if idc.GetMnem(curEA) in config_for_feature.Gemini_arithmeticInstr:
			arithNum = arithNum + 1
		#  逻辑指令
		if idc.GetMnem(curEA) in config_for_feature.Gemini_logicInstr:
			logicNum = logicNum + 1

		curEA = idc.NextHead(curEA,block.endEA)

	fea_str = str(numericNum) + ","+str(stringNum) + ","+str(transferNum) + ","+str(callNum) + ","+str(InstrNum) + ","+str(arithNum) + ","+str(logicNum) + ","
	return fea_str
def calBasicBlockFeature_vulseeker(block):
	StackNum = 0  # stackInstr
	MathNum = 0	 # arithmeticInstr
	LogicNum = 0  # logicInstr
	CompareNum = 0	# compareInstr
	ExCallNum = 0  # externalInstr
	InCallNum = 0  # internalInstr
	ConJumpNum = 0	# conditionJumpInstr
	UnConJumpNum = 0  # unconditionJumpInstr
	GeneicNum = 0  # genericInstr
	curEA = block.startEA
	while curEA <= block.endEA :
		inst = idc.GetMnem(curEA)
		if inst in config_for_feature.VulSeeker_stackInstr:
			StackNum = StackNum + 1
		elif inst in config_for_feature.VulSeeker_arithmeticInstr:
			MathNum = MathNum + 1
		elif inst in config_for_feature.VulSeeker_logicInstr:
			LogicNum = LogicNum + 1
		elif inst in config_for_feature.VulSeeker_compareInstr:
			CompareNum = CompareNum + 1
		elif inst in config_for_feature.VulSeeker_externalInstr:
			ExCallNum = ExCallNum + 1
		elif inst in config_for_feature.VulSeeker_internalInstr:
			InCallNum = InCallNum + 1
		elif inst in config_for_feature.VulSeeker_conditionJumpInstr:
			ConJumpNum = ConJumpNum + 1
		elif inst in config_for_feature.VulSeeker_unconditionJumpInstr:
			UnConJumpNum = UnConJumpNum + 1
		else:
			GeneicNum = GeneicNum + 1

		curEA = idc.NextHead(curEA,block.endEA)
		# elif inst in genericInstr:
		#	  GeneicNum = GeneicNum + 1
		# else:
		#	  print() "+++++++++", inst.insn.mnemonic,
	fea_str =  str(StackNum) + "," + str(MathNum) + "," + str(LogicNum) + "," + str(CompareNum) + "," \
			  + str(ExCallNum) + "," + str(ConJumpNum) + "," + str(UnConJumpNum) + "," + str(GeneicNum) + ","
	return fea_str

def block_fea(allblock):
	fea_str_ge=""
	fea_str_vul=""
	for block in allblock:
		gemini_str = calBasicBlockFeature_gemini(block)
		vulseeker_str = calBasicBlockFeature_vulseeker(block)
		fea_str_ge =fea_str_ge+ str(hex(block.startEA)) + "," + gemini_str + "#"
		fea_str_vul =fea_str_vul+str(hex(block.startEA)) + "," + vulseeker_str + "#"
		#fea_fp.write(fea_str)
	return fea_str_vul


def main():
	global bin_num, func_num, function_list_file, function_list_fp,functions

	fea_path_origion=""
	if len(idc.ARGV)<1:
		fea_path=config.FEA_DIR+"\\CVE-2015-1791\\DAP-1562_FIRMWARE_1.10"
		bin_path = config.O_DIR + "\\CVE-2015-1791\\DAP-1562_FIRMWARE_1.10\\wpa_supplicant.i64"
		binary_file = bin_path.split(os.sep)[-1]
		program = "CVE-2015-1791"
		version = "DAP-1562_FIRMWARE_1.10"
	else:
		print ("idc.ARGF[1]",idc.ARGV[1])
		print ("idc.ARGV[2]",idc.ARGV[2])
		fea_path_origion = idc.ARGV[1]
		fea_path_temp = idc.ARGV[1]+"\\temp"
		bin_path = idc.ARGV[2]
		binary_file = bin_path.split(os.sep)[-1]
		program = idc.ARGV[3]
		version = idc.ARGV[4]


	print ("Directory path	：", fea_path_origion)
	function_list_file = fea_path_origion + os.sep + "functions_list_fea.csv"
	function_list_fp = open(function_list_file, 'w')  # a 追加

	textStartEA = 0
	textEndEA = 0
	for seg in idautils.Segments():
		if (idc.SegName(seg)==".text"):
			textStartEA = idc.SegStart(seg)
			textEndEA = idc.SegEnd(seg)
			break


	# 遍历文件中的所有指令，保存到文件
	# 生成dict，将指令地址与指令id一一对应
	print ("遍历所有指令，生成inst_Dict, inst_info")

	novalid_name=0#修改不合法路径的名字

	##过滤掉的函数及其作为路径不合法的函数名字
	#E:\VulSeeker2\fig
	#func_filters= open("E:\\VulSeeker2\\fig\\fun-filters.csv",mode='a+')
	#basic_block_file=open("E:\\VulSeeker2\\fig\\basic_blocks.csv",mode='a+')
	#CCorpus_fuctions = open("E:\\VulSeeker2\\fig\\corpus_fuctions.csv", mode='a+')
	#function_retain=open("E:\\VulSeeker2\\fig\\function-retain.csv",mode='a+')
	#count_file = open("E:\\VulSeeker2\\fig\\count-file.txt", mode='a+')

	# count_file.write('0')
	# count = count_file.readlines()
	# Count=int(count[0])
	# print(int(count[0]))
	print("s")
	#corpus_basic_blockstring="E:\\VulSeeker2\\fig\\corpus_basic_blocks"+str(Count/3000000)+".csv"
	corpus_basic_blockstring = "E:\\VulSeeker2\\fig\\corpus_basic_blocks__bigParts.csv"



	with open(corpus_basic_blockstring.decode("utf-8"),"wb") as corpus_basic_block:


		#filename = "E:\\VulSeeker2\\fig\\dataset_small_test_file\\dataset_small_test".json"
		filename = "E:\\VulSeeker2\\fig\\dataset_small_train_file\\dataset_small_train.json"


		aaa = open('E:\\VulSeeker2\\fig\\Count', mode='w')

		aaa.close()

		with open(filename,mode='a+') as file_obj:   ##每次存进一个二进制的所有函数
			numbers = json.load(file_obj)

		for func in idautils.Functions(textStartEA, textEndEA):
			print("s121")
			# Count = Count + 1
			# Ignore Library Code --忽视程序库代码
			flags = idc.GetFunctionFlags(func)
			print("flag: ", flags)
			if flags & idc.FUNC_LIB:
				#func_filters.write(hex(func)+"--FUNC_LIB--过滤掉的函数："+idc.GetFunctionName(func)+"\n")
				continue
			if flags & idc.FUNC_THUNK:
				#func_filters.write(hex(func)+"--FUNC_THUNK--过滤掉的函数："+idc.GetFunctionName(func)+"\n")
				continue
			#function_retain.write("---retain-Functions:  ")
			#function_retain.write(str(idc.GetFunctionName(func))+'\n')
			cur_function_name = idc.GetFunctionName(func)
			print(cur_function_name)

			# if cur_function_name != "X509_NAME_get_text_by_NID":
			#	 continue

			fea_path = fea_path_origion
			if cur_function_name.lower() in functions:
				fea_path = fea_path_temp
				if not os.path.exists(fea_path):
					os.mkdir(fea_path)
			# cur_function_name = cur_function_name + "_"+time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
			functions.append(cur_function_name.lower())
			print (cur_function_name, "=====start")  # 打印函数名

			'''	  
                记录函数的控制流信息,生成CFG邻接表
                每个txt中存放一个函数的控制流图, 命名方式:[函数名_cfg.txt]
                # a b c	 # a-b a-c
                # d e  # d-e
                # G = nx.read_adjlist(‘test.adjlist’)
            '''
			allblock = idaapi.FlowChart(idaapi.get_func(func))
			cfg_file = fea_path + os.sep + str(cur_function_name) + "_cfg.txt"

			import re

			# func
			def isRulePath(file_path):
				re_path = r'^(?P<path>(?:[a-zA-Z]:)?\\(?:[^\\\?\/\*\|<>:"]+\\)+)' \
						  r'(?P<filename>(?P<name>[^\\\?\/\*\|<>:"]+?)\.' \
						  r'(?P<ext>[^.\\\?\/\*\|<>:"]+))$'
				path_flag = re.search(re_path, file_path)
				if path_flag:
					print(path_flag.group())
					rule_path = path_flag.group()
					return rule_path
				else:
					print("Invalid path")
					return False

			print(isRulePath(cfg_file))
			if isRulePath(cfg_file) == False:
				# func_filters.write("作为路径不合法的函数名字："+str(novalid_name)+cur_function_name+'\n')
				cur_function_name = str(novalid_name)
				cfg_file = fea_path + os.sep + str(cur_function_name) + "_cfg.txt"

			#cfg_fp = open(cfg_file, 'w')

			block_items = []
			DG = nx.DiGraph()
			# basic_block_file.write("basic block instrution of function: "+ cur_function_name+"\n")
			instfuc = ''
			instblocks=[]
			sigblocks=[]
			for idaBlock in allblock:
				instblock=''
				temp_str = str(hex(idaBlock.startEA))
				block_items.append(temp_str[2:])

				# basic_block_file.write("basci block start---\n")
				# basic_block_file.write(temp_str[2:]+'\n')

				curEA = idaBlock.startEA

				# basic_block_file.write("basic block end---\n")
				DG.add_node(hex(idaBlock.startEA))
				for succ_block in idaBlock.succs():
					DG.add_edge(hex(idaBlock.startEA), hex(succ_block.startEA))
				for pred_block in idaBlock.preds():
					DG.add_edge(hex(pred_block.startEA), hex(idaBlock.startEA))

				while curEA <= idaBlock.endEA:
					# inst = idc.GetDisasm(curEA)
					inst1 = ''
					inst = str(idc.GetMnem(curEA))
					a = str(idc.GetOpnd(curEA, 0))
					inst1 = inst1 + inst + " " + a
					# inst1 = inst1 + "" + str(idc.GetOpnd(curEA, 1))
					if str(idc.GetOpnd(curEA, 1)) == '':
						inst1 = inst1 + '.'
					elif str(idc.GetOpnd(curEA, 2)) == '':
						b = str(idc.GetOpnd(curEA, 1))
						inst1 = inst1 + "," + b
						inst1 = inst1 + "."
					else:
						c = str(idc.GetOpnd(curEA, 1))
						d = str(idc.GetOpnd(curEA, 2))
						inst1 = inst1 + ',' + c
						inst1 = inst1 + ',' + d
						inst1 = inst1 + '.'

					# inst = inst + "  " + str(idc.GetOpnd(curEA, 0))
					# inst = inst + " " + str(idc.GetOpnd(curEA, 1))
					# inst = inst + " " + str(idc.GetOpnd(curEA, 2))
					# inst = inst + " " + str(idc.GetOpnd(curEA, 3))
					# basic_block_file.write(str(inst)+'\n')

					curEA = idc.NextHead(curEA, idaBlock.endEA)
					print("inst1: ",inst1)

					instblock =instblock+normalization(str(inst1).strip('.'))+'.'
					instfuc = instfuc + normalization(str(inst1).strip('.'))+'.'

				instblocks.append(instblock)
			#CCorpus=csv.writer(CCorpus_fuctions)
			aa=[[instfuc.replace('.',' ')]]
			#CCorpus.writerows(aa)




			# print() DG.edges()
			# print() block_items

			#corpus_basic_block.write(instfuc + "\n") ##写入一个函数的
			cfg_str=''
			for cfg_node in DG.nodes():
				# print() cfg_node
				cfg_str = cfg_str+str(cfg_node)
				for edge in DG.succ[cfg_node]:
					cfg_str = cfg_str + " " + edge
						# print() hex(edge.addr),
			  			# print() hex(cfg_node.addr, create_using=nx.DiGraph()),"---->",hex(edge.addr)  # 遍历所有边
				# print() "cfg_str",cfg_str
				cfg_str = cfg_str + "#"
			print(cfg_str)
				#cfg_fp.write(cfg_str)

			'''
                记录函数的数据流信息生成DFG邻接表
                每个txt中存放一个函数的数据流图, 命名方式:[函数名_dfg.txt]
                # a b c	 # a-b a-c
                # d e  # d-e
                # G = nx.read_adjlist(‘test.adjlist’)
            '''


			'''
                记录函数的基本块信息,抽取一个函数中各个基本块的特征
                每个函数保存成一个CSV文件, 命名方式:[函数名_fea.csv]
                #	堆栈、算术、逻辑、比较、外部调用、内部调用、条件跳转、非条件跳转、普通指令
            '''
			fea_file = fea_path + os.sep + str(cur_function_name)+ "_fea.csv"
			if isRulePath(fea_file)==False:
				fea_file = fea_path + os.sep + str(novalid_name) + "_fea.txt"
				novalid_name = novalid_name + 1
			#fea_fp = open(fea_file, 'a+')
			fea_vul_str=block_fea(allblock)

			print("fea: ",fea_path)
			Optimi=''
			for i in str(idc.ARGV[2]).split('-'):
				if i in youhua:
					Optimi=i
					break
			#orig_file = fea_path + os.sep + str(cur_function_name)+'-'+you+ "_oi_info.csv"
			# print("len(orig_file: ",len(orig_file))
			# print(orig_file)
			print("cur_function_name: ",cur_function_name)
			print("Optimi:",Optimi)


			youxianji=Optimi ##优化等级

			cfg_str = cfg_str.strip('#')
			fea_vul_str = fea_vul_str.strip(',#')

			fea_vul_str = fea_vul_str.split(',#')
			print("fea_vul_str",fea_vul_str)
			lens = len(fea_vul_str)
			print(lens)
			fea = []
			cfg = []
			for i in range(0, lens):
				cfg.append([])
			d = dict()
			tecount = 0
			for i in fea_vul_str:
				d[i.split(',')[0]] = tecount
				dd = i.split(',')[1:]
				fea.append(dd)
				tecount = tecount + 1
			# print(i)
			print(d)
			print(str(cfg_str).split('#'))
			for j in str(cfg_str).split('#'):
				print(j)
				k = j.split(' ')
				print(k)
				for elem in range(1, len(k)):
					cfg[d[k[0]]].append(d[k[elem]])

			# B=[]
			B = [instfuc, cfg, fea,instblocks]

			if numbers.has_key(cur_function_name) == True:

				numbers[cur_function_name][youxianji] = B
			else:
				A = {}
				A[youxianji] = B
				numbers[cur_function_name] = A

			# headers =['','fuc_name','fuc_assembly']
			#
			# f_csv = csv.writer(corpus_basic_block)
			# #f_csv.writerow(headers)
			# if not os.path.exists(corpus_basic_blockstring):
			# 	f_csv.writerow(headers)
			# #rows = [[cur_function_name,instfuc]]
			# rows = [  #
			# 	['',cur_function_name+'-'+str(idc.ARGV[2]).split('-')[3],instfuc]
			# ]
			# f_csv.writerows(rows)
			#
			#
			# for instru in idautils.FuncItems(func):
			# 	orig_fp.write(hex(instru)+","+idc.GetDisasm(instru)+"\n")
			# 	inst_fp.write(idc.GetMnem(instru)+"\n")

			##_ZN4absl12lts_2021032414flags_internal8FlagImpl9ParseFromESt17basic_string_viewIcSt11char_traitsIcEENS1_15FlagSettingModeENS1_11ValueSourceERNSt7__
			'''	  
                记录函数概要信息，函数名，路径，基本块数量，控制流边的数量，数据流边的数量
            '''

			print (cur_function_name, "=====finish")  # 打印函数名
	#函数名,基本块数量,数据流结点数量,控制流边数量,数据流边数量,所在程序名,版本编号,所在二进制文件路径
			function_str = str(cur_function_name) + "," + str(DG.number_of_nodes()) +  "," + \
							   str(DG.number_of_edges()) + ","  +  "," + \
							   str(program) + "," + str(version) + "," +str(bin_path) + ",\n"

			# function_str = str(cur_function_name) + "," + str(DG.number_of_nodes()) + "," + \
			# 			   str(DG.number_of_edges()) + "," + str(dfg.number_of_edges()) + "," + \
			# 			   str(program) + "," + str(version) + "," + str(bin_path) + ",\n"
			function_list_fp.write(function_str)
	# count_file.seek(0)
	#
	# count_file.truncate()
	# count_file.write(str(Count))
	# count_file.close()
		with open(filename, 'w') as file_obj:
			json.dump(numbers,file_obj,indent=1)

	# func_filters.close()
	# basic_block_file.close()
	# function_retain.close()
	corpus_basic_block.close()
	#CCorpus_fuctions.close()

#redirect output into a file, original output is the console.
def stdout_to_file(output_file_name, output_dir=None):
	if not output_dir:
		output_dir = os.path.dirname(os.path.realpath(__file__))
	output_file_path = os.path.join(output_dir, output_file_name)
	print( output_file_path)
	print( "original output start")
	# save original stdout descriptor
	orig_stdout = sys.stdout
	# create output file
	f = open(output_file_path, "w+")
	# set stdout to output file descriptor
	sys.stdout = f
	return f, orig_stdout


if __name__=='__main__':

	print("--------------gen_fea.py_begin-----------------------" + "\n")
	f, orig_stdout = stdout_to_file(
		"output_" + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + ".txt", "E:\\VulSeeker2\\fig")
	main()
	sys.stdout = orig_stdout  # recover the output to the console window
	f.close()

	idc.Exit(0)