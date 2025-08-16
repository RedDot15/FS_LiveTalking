/*
recording
https://github.com/xiangyuecn/Recorder
*/
(function (factory) {
	factory(window);
	//umd returnExports.js
	if (typeof (define) == 'function' && define.amd) {
		define(function () {
			return Recorder;
		});
	};
	if (typeof (module) == 'object' && module.exports) {
		module.exports = Recorder;
	};
}(function (window) {
	"use strict";

	var NOOP = function () { };

	var Recorder = function (set) {
		return new initFn(set);
	};
	Recorder.LM = "2023-02-01 18:05";
	var RecTxt = "Recorder";
	var getUserMediaTxt = "getUserMedia";
	var srcSampleRateTxt = "srcSampleRate";
	var sampleRateTxt = "sampleRate";
	var CatchTxt = "catch";


	// Have you enabled global microphone recording? All work is ready, just waiting to receive audio data.
	Recorder.IsOpen = function () {
		var stream = Recorder.Stream;
		if (stream) {
			var tracks = stream.getTracks && stream.getTracks() || stream.audioTracks || [];
			var track = tracks[0];
			if (track) {
				var state = track.readyState;
				return state == "live" || state == track.LIVE;
			};
		};
		return false;
	};

	/*
	The AudioContext buffer size for H5 recording. This affects the onProcess call rate during H5 recording. Compared to AudioContext.sampleRate=48000, 4096 is closer to 12 frames per second. Adjusting this parameter can produce smoother callback animations.
	Selectable values: 256, 512, 1024, 2048, 4096, 8192, or 16384.
	Note: The value should not be set too low. Starting at 2048, different browsers may not be able to keep up with the callback rate, resulting in audio quality issues.
	Generally, no adjustment is required. After adjusting, you must close the open recording session before reopening it for the effect to take effect.
	*/
	Recorder.BufferSize = 4096;
	// Destroy all global resources held. This method needs to be called explicitly when you want to completely remove the Recorder.
	Recorder.Destroy = function () {
		CLog(RecTxt + " Destroy");
		Disconnect(); // Disconnect any global streams and resources that may exist

		for (var k in DestroyList) {
			DestroyList[k]();
		};
	};
	var DestroyList = {};
	// Register a processing method that needs to destroy global resources
	Recorder.BindDestroy = function (key, call) {
		DestroyList[key] = call;
	};

	//Determine whether the browser supports recording and can be called at any time. Note: This only detects browser support and does not determine or call user authorization, nor does it determine whether specific recording formats are supported.
	Recorder.Support = function () {
		var scope = navigator.mediaDevices || {};
		if (!scope[getUserMediaTxt]) {
			scope = navigator;
			scope[getUserMediaTxt] || (scope[getUserMediaTxt] = scope.webkitGetUserMedia || scope.mozGetUserMedia || scope.msGetUserMedia);
		};
		if (!scope[getUserMediaTxt]) {
			return false;
		};
		Recorder.Scope = scope;

		if (!Recorder.GetContext()) {
			return false;
		};
		return true;
	};

	// Get the global AudioContext object. If the browser does not support it, it will return null.
	Recorder.GetContext = function () {
		var AC = window.AudioContext;
		if (!AC) {
			AC = window.webkitAudioContext;
		};
		if (!AC) {
			return null;
		};

		if (!Recorder.Ctx || Recorder.Ctx.state == "closed") {
			// Cannot be constructed repeatedly, low version number of hardware contexts reached maximum (6)
			Recorder.Ctx = new AC();

			Recorder.BindDestroy("Ctx", function () {
				var ctx = Recorder.Ctx;
				if (ctx && ctx.close) { // If you can turn it off, turn it off. If you can't turn it off, keep it.
					ctx.close();
					Recorder.Ctx = 0;
				};
			});
		};
		return Recorder.Ctx;
	};


	/*Whether to enable MediaRecorder.WebM.PCM for audio collection connection (if the browser supports it), it is enabled by default. When disabled or not supported, AudioWorklet or ScriptProcessor will be used for connection; The audio data collected by MediaRecorder is better than other methods, with almost no frame drops, so the sound quality is obviously much better. It is recommended to keep it on.*/
	var ConnectEnableWebM = "ConnectEnableWebM";
	Recorder[ConnectEnableWebM] = true;

	/*Whether to enable the AudioWorklet feature for audio collection connection (if the browser supports it), it is disabled by default. When disabled or not supported, the outdated ScriptProcessor will be used for connection (if the method is still available). The current implementation of AudioWorklet is not as robust as ScriptProcessor on mobile devices; If ConnectEnableWebM is enabled and valid, this parameter will not take effect*/
	var ConnectEnableWorklet = "ConnectEnableWorklet";
	Recorder[ConnectEnableWorklet] = false;

	/*Initialize H5 audio collection connection. If sourceStream is provided, only a simple connection will be made. If it is a normal microphone recording, the Stream at this time is global. After disconnecting on Safari, it cannot be connected and used again, which results in silence. Therefore, all global processing is used to avoid calling disconnect; global processing also helps to shield low-level details, so there is no need to call low-level interfaces during start, which improves compatibility and reliability.*/
	var Connect = function (streamStore, isUserMedia) {
		var bufferSize = streamStore.BufferSize || Recorder.BufferSize;

		var ctx = Recorder.Ctx, stream = streamStore.Stream;
		var mediaConn = function (node) {
			var media = stream._m = ctx.createMediaStreamSource(stream);
			var ctxDest = ctx.destination, cmsdTxt = "createMediaStreamDestination";
			if (ctx[cmsdTxt]) {
				ctxDest = ctx[cmsdTxt]();
			};
			media.connect(node);
			node.connect(ctxDest);
		}
		var isWebM, isWorklet, badInt, webMTips = "";
		var calls = stream._call;

		// Audio data processing returned by the browser
		var onReceive = function (float32Arr) {
			for (var k0 in calls) {//has item
				var size = float32Arr.length;

				var pcm = new Int16Array(size);
				var sum = 0;
				for (var j = 0; j < size; j++) {//floatTo16BitPCM 
					var s = Math.max(-1, Math.min(1, float32Arr[j]));
					s = s < 0 ? s * 0x8000 : s * 0x7FFF;
					pcm[j] = s;
					sum += Math.abs(s);
				};

				for (var k in calls) {
					calls[k](pcm, sum);
				};

				return;
			};
		};

		var scriptProcessor = "ScriptProcessor";//a bunch of string names to facilitate js compression
		var audioWorklet = "audioWorklet";
		var recAudioWorklet = RecTxt + " " + audioWorklet;
		var RecProc = "RecProc";
		var MediaRecorderTxt = "MediaRecorder";
		var MRWebMPCM = MediaRecorderTxt + ".WebM.PCM";


		//===================Connection Method Three=========================
		// Antique ScriptProcessor processing, currently compatible with all browsers. Although it is an outdated method, it is more robust, and the performance on mobile devices is stronger than AudioWorklet
		var oldFn = ctx.createScriptProcessor || ctx.createJavaScriptNode;
		var oldIsBest = ". Because " + audioWorklet + " calls back 375 times a second internally, there may be performance issues on mobile devices that cause callbacks to be lost and recordings to be shortened. This does not affect PCs, so it is not recommended to enable " + audioWorklet + " for now.";
		var oldScript = function () {
			isWorklet = stream.isWorklet = false;
			_Disconn_n(stream);
			CLog("Connect uses the old " + scriptProcessor + ", " + (Recorder[ConnectEnableWorklet] ? "but it has been" : "can") + " set " + RecTxt + "." + ConnectEnableWorklet + "=true to try to enable " + audioWorklet + webMTips + oldIsBest, 3);

			var process = stream._p = oldFn.call(ctx, bufferSize, 1, 1);//mono channel, to simplify data processing
			mediaConn(process);

			var _DsetTxt = "_D220626", _Dset = Recorder[_DsetTxt]; if (_Dset) CLog("Use " + RecTxt + "." + _DsetTxt, 3);
			process.onaudioprocess = function (e) {
				var arr = e.inputBuffer.getChannelData(0);
				if (_Dset) {//temporary debugging parameter, will be removed in the future
					arr = new Float32Array(arr);//The block is shared, must be copied
					setTimeout(function () { onReceive(arr) });//Exit the callback immediately, trying to reduce the impact on browser recording
				} else {
					onReceive(arr);
				};
			};
		};


		//===================Connection Method Two=========================
		var connWorklet = function () {
			// Try to enable AudioWorklet processing
			isWebM = stream.isWebM = false;
			_Disconn_r(stream);

			isWorklet = stream.isWorklet = !oldFn || Recorder[ConnectEnableWorklet];
			var AwNode = window.AudioWorkletNode;
			if (!(isWorklet && ctx[audioWorklet] && AwNode)) {
				oldScript();//disabled or not supported, use the old one directly
				return;
			};
			var clazzUrl = function () {
				var xf = function (f) { return f.toString().replace(/^function|DEL_/g, "").replace(/\$RA/g, recAudioWorklet) };
				var clazz = 'class ' + RecProc + ' extends AudioWorkletProcessor{';
				clazz += "constructor " + xf(function (option) {
					DEL_super(option);
					var This = this, bufferSize = option.processorOptions.bufferSize;
					This.bufferSize = bufferSize;
					This.buffer = new Float32Array(bufferSize * 2);//arbitrarily giving a size messes up the buffer regardless
					This.pos = 0;
					This.port.onmessage = function (e) {
						if (e.data.kill) {
							This.kill = true;
							console.log("$RA kill call");
						}
					};
					console.log("$RA .ctor call", option);
				});

				//https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletProcessor/process Each callback returns 128 samples of data, 375 callbacks per second, high frequency leads to performance issues on mobile devices, resulting in missing callbacks and shorter recordings. There seems to be no performance issue on PCs
				clazz += "process " + xf(function (input, b, c) {//Need to wait until ctx is active to get callbacks
					var This = this, bufferSize = This.bufferSize;
					var buffer = This.buffer, pos = This.pos;
					input = (input[0] || [])[0] || [];
					if (input.length) {
						buffer.set(input, pos);
						pos += input.length;

						var len = ~~(pos / bufferSize) * bufferSize;
						if (len) {
							this.port.postMessage({ val: buffer.slice(0, len) });

							var more = buffer.subarray(len, pos);
							buffer = new Float32Array(bufferSize * 2);
							buffer.set(more);
							pos = more.length;
							This.buffer = buffer;
						}
						This.pos = pos;
					}
					return !This.kill;
				});
				clazz += '}'
					+ 'try{'
					+ 'registerProcessor("' + RecProc + '", ' + RecProc + ')'
					+ '}catch(e){'
					+ 'console.error("' + recAudioWorklet + ' registration failed",e)'
					+ '}';
				//URL.createObjectURL reports "Not allowed to load local resource" in some local browsers, so use dataurl directly
				return "data:text/javascript;base64," + btoa(unescape(encodeURIComponent(clazz)));
			};

			var awNext = function () {//can continue, disconnect was not called
				return isWorklet && stream._na;
			};
			var nodeAlive = stream._na = function () {
				//will be called at start. If no data is received, it is assumed that AudioWorklet has a problem and will fall back to the old one
				if (badInt !== "") {//no data has been called back
					clearTimeout(badInt);
					badInt = setTimeout(function () {
						badInt = 0;
						if (awNext()) {
							CLog(audioWorklet + " did not return any audio, reverting to " + scriptProcessor, 3);
							oldFn && oldScript();//In the future, there will be no old ones, which may be a misjudgment
						};
					}, 500);
				};
			};
			var createNode = function () {
				if (!awNext()) return;
				var node = stream._n = new AwNode(ctx, RecProc, {
					processorOptions: { bufferSize: bufferSize }
				});
				mediaConn(node);
				node.port.onmessage = function (e) {
					if (badInt) {
						clearTimeout(badInt); badInt = "";
					};
					if (awNext()) {
						onReceive(e.data.val);
					} else if (!isWorklet) {
						CLog(audioWorklet + " redundant callback", 3);
					};
				};
				CLog("Connect uses " + audioWorklet + ", setting " + RecTxt + "." + ConnectEnableWorklet + "=false can revert to the old " + scriptProcessor + webMTips + oldIsBest, 3);
			};

			//If resume at start and creating the node below happen at the same time, it will cause some browsers to crash. ztest_chrome_bug_AudioWorkletNode.html in the source assets can be used for testing. Therefore, wrap all code in resume (regardless of catch) to avoid this problem
			ctx.resume()[calls && "finally"](function () {//Comment out this line to see browser crash STATUS_ACCESS_VIOLATION
				if (!awNext()) return;
				if (ctx[RecProc]) {
					createNode();
					return;
				};
				var url = clazzUrl();
				ctx[audioWorklet].addModule(url).then(function (e) {
					if (!awNext()) return;
					ctx[RecProc] = 1;
					createNode();
					if (badInt) {//restart the timer
						nodeAlive();
					};
				})[CatchTxt](function (e) { //fix keyword to keep the string form when catching
					CLog(audioWorklet + ".addModule failed", 1, e);
					awNext() && oldScript();
				});
			});
		};


		//===================Connection Method One=========================
		var connWebM = function () {
			// Try to enable MediaRecorder for webm+pcm recording processing
			var MR = window[MediaRecorderTxt];
			var onData = "ondataavailable";
			var webmType = "audio/webm; codecs=pcm";
			isWebM = stream.isWebM = Recorder[ConnectEnableWebM];

			var supportMR = MR && (onData in MR.prototype) && MR.isTypeSupported(webmType);
			webMTips = supportMR ? "" : "(This browser does not support " + MRWebMPCM + ")";
			if (!isUserMedia || !isWebM || !supportMR) {
				connWorklet(); //non-microphone recording (MediaRecorder sample rate is uncontrollable) or disabled or MediaRecorder not supported or webm+pcm not supported
				return;
			}

			var mrNext = function () {//can continue, disconnect was not called
				return isWebM && stream._ra;
			};
			var mrAlive = stream._ra = function () {
				//will be called at start. If no data is received, it is assumed that MediaRecorder has a problem and will be downgraded
				if (badInt !== "") {//no data has been called back
					clearTimeout(badInt);
					badInt = setTimeout(function () {
						//badInt=0; keep it for nodeAlive to continue judging
						if (mrNext()) {
							CLog(MediaRecorderTxt + " did not return any audio, downgrading to " + audioWorklet, 3);
							connWorklet();
						};
					}, 500);
				};
			};

			var mrSet = Object.assign({ mimeType: webmType }, Recorder.ConnectWebMOptions);
			var mr = stream._r = new MR(stream, mrSet);
			var webmData = stream._rd = { sampleRate: ctx[sampleRateTxt] };
			mr[onData] = function (e) {
				//extract pcm data from webm, if extraction fails, wait for badInt timeout and downgrade
				var reader = new FileReader();
				reader.onloadend = function () {
					if (mrNext()) {
						var f32arr = WebM_Extract(new Uint8Array(reader.result), webmData);
						if (!f32arr) return;
						if (f32arr == -1) {//unable to extract, downgrade immediately
							connWorklet();
							return;
						};

						if (badInt) {
							clearTimeout(badInt); badInt = "";
						};
						onReceive(f32arr);
					} else if (!isWebM) {
						CLog(MediaRecorderTxt + " redundant callback", 3);
					};
				};
				reader.readAsArrayBuffer(e.data);
			};
			mr.start(~~(bufferSize / 48));//callback interval based on 48k
			CLog("Connect uses " + MRWebMPCM + ", setting " + RecTxt + "." + ConnectEnableWebM + "=false can revert to using " + audioWorklet + " or the old " + scriptProcessor);
		};

		connWebM();
	};
	var ConnAlive = function (stream) {
		if (stream._na) stream._na(); //Check if the AudioWorklet connection is valid, if not, roll back to the old ScriptProcessor
		if (stream._ra) stream._ra(); //Check if the MediaRecorder connection is valid, if not, downgrade
	};
	var _Disconn_n = function (stream) {
		stream._na = null;
		if (stream._n) {
			stream._n.port.postMessage({ kill: true });
			stream._n.disconnect();
			stream._n = null;
		};
	};
	var _Disconn_r = function (stream) {
		stream._ra = null;
		if (stream._r) {
			stream._r.stop();
			stream._r = null;
		};
	};
	var Disconnect = function (streamStore) {
		streamStore = streamStore || Recorder;
		var isGlobal = streamStore == Recorder;

		var stream = streamStore.Stream;
		if (stream) {
			if (stream._m) {
				stream._m.disconnect();
				stream._m = null;
			};
			if (stream._p) {
				stream._p.disconnect();
				stream._p.onaudioprocess = stream._p = null;
			};
			_Disconn_n(stream);
			_Disconn_r(stream);

			if (isGlobal) {//In the global context, the stream (microphone) needs to be turned off. Streams provided directly are not processed
				var tracks = stream.getTracks && stream.getTracks() || stream.audioTracks || [];
				for (var i = 0; i < tracks.length; i++) {
					var track = tracks[i];
					track.stop && track.stop();
				};
				stream.stop && stream.stop();
			};
		};
		streamStore.Stream = 0;
	};

	/*Convert the sample rate of pcm data
	pcmDatas: [[Int16,...]] list of pcm fragments
	pcmSampleRate:48000 pcm data sample rate
	newSampleRate:16000 sample rate to be converted to. If newSampleRate>=pcmSampleRate, no processing will be performed. If it is smaller, it will be resampled
	prevChunkInfo:{} optional, return value of the last call, used for continuous conversion. This call will start processing from the end position of the last one. Or you can define a ChunkInfo yourself to start conversion from a specific position in pcmDatas
	option:{ optional, configuration items
		frameSize:123456 frame size, the number of PCM Int16s per frame. The length of the pcm after sample rate conversion will be an integer multiple of frameSize, used for continuous conversion. Currently only useful for mp3 format, where frameSize is 1152, so the duration of the encoded mp3 is exactly the same as the duration of the pcm. Otherwise, the mp3 duration will be longer because padding data is added when the last frame of the recording is not full.
		frameType:"" frame type, generally rec.set.type. When this parameter is provided, there is no need to provide frameSize, and the best value for frameSize will be automatically used. Currently only supports mp3=1152 (number of samples per frame in MPEG1 Layer3), other types=1.
			The above two parameters are used for continuous conversion, at most one can be used. If not provided, no special frame processing will be performed. If provided, prevChunkInfo must also be provided to be effective. When processing the last segment of data, there is no need to provide the frame size to output the last tiny bit of remaining data.
	}
	
	Return ChunkInfo:{
		//can be defined, starts conversion from the specified position to the end
		index:0 index of pcmDatas that has been processed
		offset:0.0 the next position of the offset in the pcm corresponding to the processed index
	    
		//only as return value
		frameNext:null||[Int16,...] partial data of the next frame, may exist when frameSize is set
		sampleRate:16000 resulting sample rate, <=newSampleRate
		data:[Int16,...] converted PCM result; If it is a continuous conversion and there is no new data in pcmDatas, the length of data may be 0
	}
	*/
	Recorder.SampleData = function (pcmDatas, pcmSampleRate, newSampleRate, prevChunkInfo, option) {
		prevChunkInfo || (prevChunkInfo = {});
		var index = prevChunkInfo.index || 0;
		var offset = prevChunkInfo.offset || 0;

		var frameNext = prevChunkInfo.frameNext || [];
		option || (option = {});
		var frameSize = option.frameSize || 1;
		if (option.frameType) {
			frameSize = option.frameType == "mp3" ? 1152 : 1;
		};

		var nLen = pcmDatas.length;
		if (index > nLen + 1) {
			CLog("SampleData seems to have passed an unreset chunk " + index + ">" + nLen, 3);
		};
		var size = 0;
		for (var i = index; i < nLen; i++) {
			size += pcmDatas[i].length;
		};
		size = Math.max(0, size - Math.floor(offset));

		//sampling https://www.cnblogs.com/blqw/p/3782420.html
		var step = pcmSampleRate / newSampleRate;
		if (step > 1) {//new sample rate is lower than recording sample rate, perform downsampling
			size = Math.floor(size / step);
		} else {//new sample rate is higher than recording sample rate, no processing, which saves interpolation processing
			step = 1;
			newSampleRate = pcmSampleRate;
		};

		size += frameNext.length;
		var res = new Int16Array(size);
		var idx = 0;
		//add the remaining data from the last time that was not enough for one frame
		for (var i = 0; i < frameNext.length; i++) {
			res[idx] = frameNext[i];
			idx++;
		};
		//process data
		for (; index < nLen; index++) {
			var o = pcmDatas[index];
			var i = offset, il = o.length;
			while (i < il) {
				//res[idx]=o[Math.round(i)]; simple downsampling directly

				//https://www.cnblogs.com/xiaoqi/p/6993912.html
				//current point = current point + increment to the next point. The sound quality is slightly better than simple downsampling
				var before = Math.floor(i);
				var after = Math.ceil(i);
				var atPoint = i - before;

				var beforeVal = o[before];
				var afterVal = after < il ? o[after]
					: (//the next point is out of bounds, look up the next array
						(pcmDatas[index + 1] || [beforeVal])[0] || 0
					);
				res[idx] = beforeVal + (afterVal - beforeVal) * atPoint;

				idx++;
				i += step;//downsampling
			};
			offset = i - il;
		};
		//frame processing
		frameNext = null;
		var frameNextSize = res.length % frameSize;
		if (frameNextSize > 0) {
			var u8Pos = (res.length - frameNextSize) * 2;
			frameNext = new Int16Array(res.buffer.slice(u8Pos));
			res = new Int16Array(res.buffer.slice(0, u8Pos));
		};

		return {
			index: index
			, offset: offset

			, frameNext: frameNext
			, sampleRate: newSampleRate
			, data: res
		};
	};


	/* A method to calculate the volume percentage
	pcmAbsSum: the sum of the absolute values of all samples in pcm Int16
	pcmLength: the length of the pcm
	Return value: 0-100, mainly used as a percentage
	Note: This is not decibels, so the name volume is not used*/
	Recorder.PowerLevel = function (pcmAbsSum, pcmLength) {
		/*Calculate volume https://blog.csdn.net/jody1989/article/details/73480259
		Higher sensitivity algorithm:
			Limit maximum sense value to 10000
				Linear curve: not friendly to low volume
					power/10000*100 
				Logarithmic curve: friendly to low volume, but requires a minimum sense value limit
					(1+Math.log10(power/10000))*100
		*/
		var power = (pcmAbsSum / pcmLength) || 0;//NaN
		var level;
		if (power < 1251) {//the result for 1250 is 10%, smaller volumes use linear values
			level = Math.round(power / 1250 * 10);
		} else {
			level = Math.round(Math.min(100, Math.max(0, (1 + Math.log(power / 10000) / Math.log(10)) * 100)));
		};
		return level;
	};

	/*Calculate volume, unit dBFS (relative level of full scale)
	maxSample: the largest absolute value of a 16-bit pcm sample (to calculate peak volume), or the average of the absolute values of all samples in the pcm
	Return value: -100~0 (maximum 0dB, minimum -100 replaces -∞)
	*/
	Recorder.PowerDBFS = function (maxSample) {
		var val = Math.max(0.1, maxSample || 0), Pref = 0x7FFF;
		val = Math.min(val, Pref);
		//https://www.logiclocmusic.com/can-you-tell-the-decibel/
		//https://blog.csdn.net/qq_17256689/article/details/120442510
		val = 20 * Math.log(val / Pref) / Math.log(10);
		return Math.max(-100, Math.round(val));
	};




	// Log output with time, can be set to an empty function to block log output
	// CLog(msg,errOrLogMsg, logMsg...) err is a number representing the log type 1:error 2:log default 3:warn, otherwise it is treated as content output. The first parameter cannot be an object because time needs to be concatenated, and an unlimited number of output parameters can follow
	Recorder.CLog = function (msg, err) {
		var now = new Date();
		var t = ("0" + now.getMinutes()).substr(-2)
			+ ":" + ("0" + now.getSeconds()).substr(-2)
			+ "." + ("00" + now.getMilliseconds()).substr(-3);
		var recID = this && this.envIn && this.envCheck && this.id;
		var arr = ["[" + t + " " + RecTxt + (recID ? ":" + recID : "") + "]" + msg];
		var a = arguments, console = window.console || {};
		var i = 2, fn = console.log;
		if (typeof (err) == "number") {
			fn = err == 1 ? console.error : err == 3 ? console.warn : fn;
		} else {
			i = 1;
		};
		for (; i < a.length; i++) {
			arr.push(a[i]);
		};
		if (IsLoser) {//Antique browser, only guarantee basic executable code and no exceptions
			fn && fn("[IsLoser]" + arr[0], arr.length > 1 ? arr : "");
		} else {
			fn.apply(console, arr);
		};
	};
	var CLog = function () { Recorder.CLog.apply(this, arguments); };
	var IsLoser = true; try { IsLoser = !console.log.apply; } catch (e) { };




	var ID = 0;
	function initFn(set) {
		this.id = ++ID;

		//If traffic statistics are enabled, an image request will be sent here
		Traffic();


		var o = {
			type: "mp3" // Output type: mp3, wav. Wav output file size is too large, not recommended. But mp3 encoding support will make the js file very large. If mp3 support is not needed, the js file can be significantly reduced
			, bitRate: 16 // Bitrate. wav: 16 or 8 bits. MP3: 8kbps 1k/s, 8kbps 2k/s, the recording file is very small

			, sampleRate: 16000 // Sample rate. wav format size = sampleRate * time. For mp3, this affects low bitrates, but has almost no effect on high bitrates.
			// wav can be any value, mp3 values: 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000
			// For sample rate reference: https://www.cnblogs.com/devin87/p/mp3-recorder.html

			, onProcess: NOOP // fn(buffers,powerLevel,bufferDuration,bufferSampleRate,newBufferIdx,asyncEnd) buffers=[[Int16,...],...]: Buffered PCM data, all pcm fragments from the start of recording to now; powerLevel: current buffer volume level 0-100; bufferDuration: buffered duration; bufferSampleRate: sample rate used for buffering (when the type supports real-time encoding (Worker), this sample rate is the same as the set sample rate, otherwise it is not necessarily the same); newBufferIdx: starting index of the newly added buffer in this callback; asyncEnd: fn() If onProcess is asynchronous (when the return value is true), this callback needs to be called after processing is complete. If it is not asynchronous, ignore this parameter. This callback must be truly asynchronous (if not truly asynchronous, it needs to be wrapped in setTimeout). onProcess return value: If true is returned, it means that asynchronous mode is enabled. Asynchrony is necessary for some computationally intensive tasks. asyncEnd must be called after asynchronous processing is complete (if not truly asynchronous, it needs to be wrapped in setTimeout). After onProcess is executed, all newly added buffers will be replaced with empty arrays, so at the beginning of this callback, all buffers from newBufferIdx to the end of this callback should be immediately saved to another array, and then written back to the end position of this callback in buffers after processing is complete.

			//*******Advanced settings******
			//,sourceStream:MediaStream Object
			// Optional, directly provide a media stream, and record and process audio data from this stream in real time (this Recorder instance has exclusive access to this stream); If not provided, it is a normal microphone recording, and the audio stream is provided by getUserMedia (all Recorder instances share the same stream)
			// For example: the stream returned by the captureStream method of audio and video tag DOM nodes (experimental feature, not highly supported by different browsers); remote streams in WebRTC; self-created streams, etc.
			// Note: The stream must contain at least one audio track. For example, an audio tag must wait until it can start playing before it has an audio track, otherwise open will fail

			//,audioTrackSet:{ deviceId:"",groupId:"", autoGainControl:true, echoCancellation:true, noiseSuppression:true }
			// audio configuration parameters for the getUserMedia method for normal microphone recording, such as specifying device ID, echo cancellation, and noise suppression switches; Note: any provided configuration value may not take effect
			// Since the microphone is globally shared, you need to close the previous one and reopen it after a new configuration
			// For more reference: https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackConstraints

			//,disableEnvInFix:false internal parameter, disables the audio input loss compensation function when the device is stuck

			//,takeoffEncodeChunk:NOOP //fn(chunkBytes) chunkBytes=[Uint8,...]: Take over the encoder output in a real-time encoding environment. This method is called in real time when the encoder encodes an effective binary audio data chunk. The parameter is a binary Uint8Array, which is the audio data fragment encoded, and all chunkBytes concatenated together is the complete audio. The idea for this implementation was originally proposed by QQ2543775048
			// When this callback method is provided, the data output of the encoder will be taken over, and the encoder will give up storing the generated audio data internally; The environment requirements are more stringent: if the current environment does not support real-time encoding processing, the fail logic will be executed directly when open is called
			// Therefore, after providing this callback, calling the stop method will not get valid audio data, because there is no audio data in the encoder. Therefore, the blob returned by stop will be a blob with a byte length of 0
			// Currently, only the mp3 format has real-time encoding implemented. In an environment that supports real-time processing, the encoded mp3 fragments will be called back in real time through this method. All chunkBytes concatenated together is the complete mp3. The sound quality of this concatenated result is better than that of the real-time generation of the mock method, because it naturally avoids the silence at the beginning and end
			// Currently, other formats besides mp3 cannot provide this callback. If provided, the fail logic will be executed directly when open is called
		};

		for (var k in set) {
			o[k] = set[k];
		};
		this.set = o;

		this._S = 9;//stop sync lock, stop can prevent start that has not yet run during the open process
		this.Sync = { O: 9, C: 9 };//same as Recorder.Sync, but this is not global, only used to simplify code logic, no actual effect
	};
	// Sync lock, controls competition for the Stream; used to interrupt an asynchronous open during close; if an object's open changes, close must be prevented, and the control of the Stream is handed over to the new object
	Recorder.Sync = {/*open*/O: 9,/*close*/C: 9 };

	Recorder.prototype = initFn.prototype = {
		CLog: CLog

		// Where the stream-related data is stored; if sourceStream is provided, the data is stored directly in the current object, otherwise it is stored globally
		, _streamStore: function () {
			if (this.set.sourceStream) {
				return this;
			} else {
				return Recorder;
			}
		}

		// Open the recording resource True(), False(msg,isUserNotAllow), need to call close. Note: this method is asynchronous; it is generally opened when in use and closed immediately after use; it can be called repeatedly to test whether recording is possible
		, open: function (True, False) {
			var This = this, streamStore = This._streamStore();
			True = True || NOOP;
			var failCall = function (errMsg, isUserNotAllow) {
				isUserNotAllow = !!isUserNotAllow;
				This.CLog("Recording open failed: " + errMsg + ",isUserNotAllow:" + isUserNotAllow, 1);
				False && False(errMsg, isUserNotAllow);
			};

			var ok = function () {
				This.CLog("open ok id:" + This.id);
				True();

				This._SO = 0;//remove stop's prevention of start calls during open
			};


			// Sync lock
			var Lock = streamStore.Sync;
			var lockOpen = ++Lock.O, lockClose = Lock.C;
			This._O = This._O_ = lockOpen;//remember the current open, if it changes, close should be prevented. This assumes the new object has replaced the current one and is no longer in use
			This._SO = This._S;//remember stop during the open process. After any stop call in the middle, start in open cannot be continued
			var lockFail = function () {
				// Multiple opens are allowed, but no closes are allowed, or the object itself has called close
				if (lockClose != Lock.C || !This._O) {
					var err = "open cancelled";
					if (lockOpen == Lock.O) {
						// no new open, close has been called to cancel, at this point the last close should take effect
						This.close();
					} else {
						err = "open interrupted";
					};
					failCall(err);
					return true;
				};
			};

			// Environment configuration check
			var checkMsg = This.envCheck({ envName: "H5", canProcess: true });
			if (checkMsg) {
				failCall("Cannot record: " + checkMsg);
				return;
			};


			//***********Audio stream has been provided directly************
			if (This.set.sourceStream) {
				if (!Recorder.GetContext()) {
					failCall("This browser does not support getting recordings from streams");
					return;
				};

				Disconnect(streamStore);//May have been opened, try to disconnect first
				This.Stream = This.set.sourceStream;
				This.Stream._call = {};

				try {
					Connect(streamStore);
				} catch (e) {
					failCall("Failed to open recording from stream: " + e.message);
					return;
				}
				ok();
				return;
			};


			//***********Open microphone to get a global audio stream************
			var codeFail = function (code, msg) {
				try {//check cross-domain first
					window.top.a;
				} catch (e) {
					failCall('No permission to record (cross-domain, please try to add a microphone access policy to the iframe, such as allow="camera;microphone")');
					return;
				};

				if (/Permission|Allow/i.test(code)) {
					failCall("User denied recording permission", true);
				} else if (window.isSecureContext === false) {
					failCall("The browser prohibits recording on insecure pages, you can solve this by enabling https");
				} else if (/Found/i.test(code)) {//May be due to no device in an insecure environment
					failCall(msg + "，No microphone available");
				} else {
					failCall(msg);
				};
			};


			// If already open and valid, don't open again
			if (Recorder.IsOpen()) {
				ok();
				return;
			};
			if (!Recorder.Support()) {
				codeFail("", "This browser does not support recording");
				return;
			};

			// Request permission, if never authorized, the browser will usually pop up a permission request box
			var f1 = function (stream) {
				// https://github.com/xiangyuecn/Recorder/issues/14 The track.readyState!="live" obtained, it may be normal at the time of callback, but may be closed after a while for unknown reasons. Delay to ensure it is truly asynchronous. This does not affect normal browsers
				setTimeout(function () {
					stream._call = {};
					var oldStream = Recorder.Stream;
					if (oldStream) {
						Disconnect(); //disconnect the existing one directly, the old Connect that has not completed will automatically terminate
						stream._call = oldStream._call;
					};
					Recorder.Stream = stream;
					if (lockFail()) return;

					if (Recorder.IsOpen()) {
						if (oldStream) This.CLog("Found multiple open calls at the same time", 1);

						Connect(streamStore, 1);
						ok();
					} else {
						failCall("Recording function is invalid: no audio stream");
					};
				}, 100);
			};
			var f2 = function (e) {
				var code = e.name || e.message || e.code + ":" + e;
				This.CLog("Request for recording permission error", 1, e);

				codeFail(code, "Cannot record: " + code);
			};

			var trackSet = {
				noiseSuppression: false //default to disable noise reduction, record original sound, to avoid strange behavior on mobile devices (including the system playing sound becoming quieter)
				, echoCancellation: false //echo cancellation
			};
			var trackSet2 = This.set.audioTrackSet;
			for (var k in trackSet2) trackSet[k] = trackSet2[k];
			trackSet.sampleRate = Recorder.Ctx.sampleRate;//must specify sample rate, otherwise MediaRecorder sample rate is 16k on mobile phones

			try {
				var pro = Recorder.Scope[getUserMediaTxt]({ audio: trackSet }, f1, f2);
			} catch (e) {//If trackSet cannot be set, just ignore it
				This.CLog(getUserMediaTxt, 3, e);
				pro = Recorder.Scope[getUserMediaTxt]({ audio: true }, f1, f2);
			};
			if (pro && pro.then) {
				pro.then(f1)[CatchTxt](f2); //fix keyword to keep the string form when catching
			};
		}
		// Close and release recording resources
		, close: function (call) {
			call = call || NOOP;

			var This = this, streamStore = This._streamStore();
			This._stop();

			var Lock = streamStore.Sync;
			This._O = 0;
			if (This._O_ != Lock.O) {
				// The control of the unique resource Stream has been handed over to the new object, so it cannot be closed here. This may leak in browsers that prompt for permission every time. A new object that is denied permission may not call close. Ignore this case and do not process it
				This.CLog("close is ignored (because multiple recs were opened at the same time, only the last one will actually close)", 3);
				call();
				return;
			};
			Lock.C++;//get control

			Disconnect(streamStore);

			This.CLog("close");
			call();
		}





		/* Simulate a piece of recording data, after which stop can be called for encoding, pcm data [1,2,3...] and pcm sample rate need to be provided*/
		, mock: function (pcmData, pcmSampleRate) {
			var This = this;
			This._stop();//clear existing resources

			This.isMock = 1;
			This.mockEnvInfo = null;
			This.buffers = [pcmData];
			This.recSize = pcmData.length;
			This[srcSampleRateTxt] = pcmSampleRate;
			return This;
		}
		, envCheck: function (envInfo) {//Availability check in the platform environment, can be called at any time to check, returns errMsg:"" for normal, "reason for failure"
			//envInfo={envName:"H5",canProcess:true}
			var errMsg, This = this, set = This.set;

			// Check the byte order of the CPU, TypedArray byte order is a mystery, directly reject the rare big-endian mode, because there is no such CPU to test
			var tag = "CPU_BE";
			if (!errMsg && !Recorder[tag] && window.Int8Array && !new Int8Array(new Int32Array([1]).buffer)[0]) {
				Traffic(tag); //If traffic statistics are enabled, an image request will be sent here
				errMsg = "Does not support " + tag + " architecture";
			};

			// The encoder checks if the configuration is available in the environment
			if (!errMsg) {
				var type = set.type;
				if (This[type + "_envCheck"]) {//The encoder has implemented the environment check
					errMsg = This[type + "_envCheck"](envInfo, set);
				} else {//Manually check if the configuration is valid for those without an implemented check
					if (set.takeoffEncodeChunk) {
						errMsg = type + " type" + (This[type] ? "" : "(encoder not loaded)") + " does not support setting takeoffEncodeChunk";
					};
				};
			};

			return errMsg || "";
		}
		, envStart: function (mockEnvInfo, sampleRate) {//platform-specific start call
			var This = this, set = This.set;
			This.isMock = mockEnvInfo ? 1 : 0;//non-H5 environment needs to enable mock and provide environment information required by envCheck
			This.mockEnvInfo = mockEnvInfo;
			This.buffers = [];//data buffer
			This.recSize = 0;//data size

			This.envInLast = 0;//time of the last recording content received by envIn
			This.envInFirst = 0;//time of the first recording content received by envIn
			This.envInFix = 0;//total compensation time
			This.envInFixTs = [];//compensation count list

			//engineCtx needs to determine the final sample rate in advance
			var setSr = set[sampleRateTxt];
			if (setSr > sampleRate) {
				set[sampleRateTxt] = sampleRate;
			} else { setSr = 0 }
			This[srcSampleRateTxt] = sampleRate;
			This.CLog(srcSampleRateTxt + ": " + sampleRate + " set." + sampleRateTxt + ": " + set[sampleRateTxt] + (setSr ? " ignore " + setSr : ""), setSr ? 3 : 0);

			This.engineCtx = 0;
			//This type supports real-time encoding (Worker)
			if (This[set.type + "_start"]) {
				var engineCtx = This.engineCtx = This[set.type + "_start"](set);
				if (engineCtx) {
					engineCtx.pcmDatas = [];
					engineCtx.pcmSize = 0;
				};
			};
		}
		, envResume: function () {//resume recording regardless of the platform environment
			//restart counting
			this.envInFixTs = [];
		}, envIn: function (pcm, sum) {//Input pcm[Int16] that is independent of the platform environment
			var This = this, set = This.set, engineCtx = This.engineCtx;
			var bufferSampleRate = This[srcSampleRateTxt];
			var size = pcm.length;
			var powerLevel = Recorder.PowerLevel(sum, size);

			var buffers = This.buffers;
			var bufferFirstIdx = buffers.length;//The previous buffers have all been processed by onProcess and are not allowed to be modified again.
			buffers.push(pcm);

			//When there is engineCtx, it will be overwritten, so save a copy here
			var buffersThis = buffers;
			var bufferFirstIdxThis = bufferFirstIdx;

			//Stuttering and loss compensation: When the device is very slow, the amount of data received by H5 is not enough, causing the playback speed to change, and the result is shorter than the actual duration. This ensures that it will not become shorter, but it cannot repair the lost audio data and the sound quality deteriorates. The current algorithm uses input time to detect whether the next frame needs to add a compensation frame. It will not start detection until (6 inputs || more than 1 second). If more than 1/3 is lost in the sliding window, compensation will be performed.
			var now = Date.now();
			var pcmTime = Math.round(size / bufferSampleRate * 1000);
			This.envInLast = now;
			if (This.buffers.length == 1) {//Record the recording time of the first recording data
				This.envInFirst = now - pcmTime;
			};
			var envInFixTs = This.envInFixTs;
			envInFixTs.splice(0, 0, { t: now, d: pcmTime });
			//Retain a 3-second sliding window for counting. In addition, pauses of more than 3 seconds are not compensated.
			var tsInStart = now, tsPcm = 0;
			for (var i = 0; i < envInFixTs.length; i++) {
				var o = envInFixTs[i];
				if (now - o.t > 3000) {
					envInFixTs.length = i;
					break;
				};
				tsInStart = o.t;
				tsPcm += o.d;
			};
			//When the required amount of data is reached, start to detect whether compensation is needed
			var tsInPrev = envInFixTs[1];
			var tsIn = now - tsInStart;
			var lost = tsIn - tsPcm;
			if (lost > tsIn / 3 && (tsInPrev && tsIn > 1000 || envInFixTs.length >= 6)) {
				//Too much loss, start compensation
				var addTime = now - tsInPrev.t - pcmTime;//Lost so many ms since the last input
				if (addTime > pcmTime / 5) {//Lost more than 1/5 of this frame
					var fixOpen = !set.disableEnvInFix;
					This.CLog("[" + now + "]" + (fixOpen ? "" : "not") + "compensating" + addTime + "ms", 3);
					This.envInFix += addTime;

					//Compensate with silence
					if (fixOpen) {
						var addPcm = new Int16Array(addTime * bufferSampleRate / 1000);
						size += addPcm.length;
						buffers.push(addPcm);
					};
				};
			};


			var sizeOld = This.recSize, addSize = size;
			var bufferSize = sizeOld + addSize;
			This.recSize = bufferSize;//This value needs to be corrected after onProcess, and the new data may be modified
			//This type has side-by-side transcoding (Worker) support, and real-time transcoding is enabled
			if (engineCtx) {
				//Convert to the sample rate of the set
				var chunkInfo = Recorder.SampleData(buffers, bufferSampleRate, set[sampleRateTxt], engineCtx.chunkInfo);
				engineCtx.chunkInfo = chunkInfo;

				sizeOld = engineCtx.pcmSize;
				addSize = chunkInfo.data.length;
				bufferSize = sizeOld + addSize;
				engineCtx.pcmSize = bufferSize;//This value needs to be corrected after onProcess, and the new data may be modified

				buffers = engineCtx.pcmDatas;
				bufferFirstIdx = buffers.length;
				buffers.push(chunkInfo.data);
				bufferSampleRate = chunkInfo[sampleRateTxt];
			};

			var duration = Math.round(bufferSize / bufferSampleRate * 1000);
			var bufferNextIdx = buffers.length;
			var bufferNextIdxThis = buffersThis.length;

			//Allow asynchronous processing of buffer data
			var asyncEnd = function () {
				//Recalculate the size. Asynchronous has already subtracted the added data. Synchronous needs to remove the added data and then recalculate.
				var num = asyncBegin ? 0 : -addSize;
				var hasClear = buffers[0] == null;
				for (var i = bufferFirstIdx; i < bufferNextIdx; i++) {
					var buffer = buffers[i];
					if (buffer == null) {//The memory has been actively released, for example, during long-term real-time transmission recording
						hasClear = 1;
					} else {
						num += buffer.length;

						//Push to the background for transcoding while recording
						if (engineCtx && buffer.length) {
							This[set.type + "_encode"](engineCtx, buffer);
						};
					};
				};

				//Synchronously clear This.buffers. No matter how many buffers are cleared, buffersThis that are not used are completely cleared.
				if (hasClear && engineCtx) {
					var i = bufferFirstIdxThis;
					if (buffersThis[0]) {
						i = 0;
					};
					for (; i < bufferNextIdxThis; i++) {
						buffersThis[i] = null;
					};
				};

				//Count the modified size. If a clear occurs asynchronously, add it back as it is. No operation is required for synchronous.
				if (hasClear) {
					num = asyncBegin ? addSize : 0;

					buffers[0] = null;//Completely cleared
				};
				if (engineCtx) {
					engineCtx.pcmSize += num;
				} else {
					This.recSize += num;
				};
			};
			//Real-time callback to process data, allowing modification or replacement of newly added data since the last callback, but not allowing modification of processed data, not allowing adding or deleting the first-level array, and allowing the second-level array to be arbitrarily modified or replaced with an empty array.
			var asyncBegin = 0, procTxt = "rec.set.onProcess";
			try {
				asyncBegin = set.onProcess(buffers, powerLevel, duration, bufferSampleRate, bufferFirstIdx, asyncEnd);
			} catch (e) {
				//Do not use CLog for this error display, so that the same content will not be printed repeatedly in the console.
				console.error(procTxt + "Callback error is not allowed, it must be guaranteed that no exception will be thrown", e);
			};

			var slowT = Date.now() - now;
			if (slowT > 10 && This.envInFirst - now > 1000) { //Start onProcess performance monitoring after 1 second
				This.CLog(procTxt + "low performance, takes" + slowT + "ms", 3);
			};

			if (asyncBegin === true) {
				//Asynchronous mode is enabled, and onProcess has taken over the new buffers data. Clear it immediately to avoid unprocessed data.
				var hasClear = 0;
				for (var i = bufferFirstIdx; i < bufferNextIdx; i++) {
					if (buffers[i] == null) {//The memory has been actively released, for example, during long-term real-time transmission recording, but asynchronous mode must be enabled. This situation is illegal.
						hasClear = 1;
					} else {
						buffers[i] = new Int16Array(0);
					};
				};

				if (hasClear) {
					This.CLog("Buffers cannot be cleared before entering asynchronous mode", 3);
				} else {
					//Restore the size. After asynchronous is finished, count only the modified size. If a clear occurs, add it back as it is.
					if (engineCtx) {
						engineCtx.pcmSize -= addSize;
					} else {
						This.recSize -= addSize;
					};
				};
			} else {
				asyncEnd();
			};
		}



		//Start recording, you need to call open first; as long as open is successful, calling this method is safe. If you force a call without open, any internal errors will not be prompted, and you will naturally get an error when you stop.
		, start: function () {
			var This = this, ctx = Recorder.Ctx;

			var isOpen = 1;
			if (This.set.sourceStream) {//A stream is provided directly, only check if open has been called
				if (!This.Stream) {
					isOpen = 0;
				}
			} else if (!Recorder.IsOpen()) {//Check if the global microphone is open and valid
				isOpen = 0;
			};
			if (!isOpen) {
				This.CLog("Not open", 1);
				return;
			};
			This.CLog("Start recording");

			This._stop();
			This.state = 3;//0 no recording 1 recording 2 paused 3 waiting for ctx activation
			This.envStart(null, ctx[sampleRateTxt]);

			//Check if stop has been called during the open process
			if (This._SO && This._SO + 1 != This._S) {//_stop was called once above
				//If stop is called before open is completed, this situation should be avoided as much as possible, and the start will be terminated.
				This.CLog("Start interrupted", 3);
				return;
			};
			This._SO = 0;

			var end = function () {
				if (This.state == 3) {
					This.state = 1;
					This.resume();
				}
			};
			if (ctx.state == "suspended") {
				var tag = "AudioContext resume: ";
				This.CLog(tag + "wait...");
				ctx.resume().then(function () {
					This.CLog(tag + ctx.state);
					end();
				})[CatchTxt](function (e) { //Less common, may not affect recording
					This.CLog(tag + ctx.state + " may not be able to record: " + e.message, 1, e);
					end();
				});
			} else {
				end();
			};
		}
		/*Pause recording*/
		, pause: function () {
			var This = this;
			if (This.state) {
				This.state = 2;
				This.CLog("pause");
				delete This._streamStore().Stream._call[This.id];
			};
		}
		/*Resume recording*/
		, resume: function () {
			var This = this;
			if (This.state) {
				This.state = 1;
				This.CLog("resume");
				This.envResume();

				var stream = This._streamStore().Stream;
				stream._call[This.id] = function (pcm, sum) {
					if (This.state == 1) {
						This.envIn(pcm, sum);
					};
				};
				ConnAlive(stream);//AudioWorklet will only run after ctx is activated
			};
		}



		, _stop: function (keepEngine) {
			var This = this, set = This.set;
			if (!This.isMock) {
				This._S++;
			};
			if (This.state) {
				This.pause();
				This.state = 0;
			};
			if (!keepEngine && This[set.type + "_stop"]) {
				This[set.type + "_stop"](This.engineCtx);
				This.engineCtx = 0;
			};
		}
		/*
		Stop recording and return the recorded data blob object
		True(blob,duration) blob: Recorded data in audio/mp3|wav format
		duration: Recording duration, in milliseconds
		False(msg)
		autoClose:false Optional, whether to automatically call close, the default is false
		*/
		, stop: function (True, False, autoClose) {
			var This = this, set = This.set, t1;
			var envInMS = This.envInLast - This.envInFirst, envInLen = envInMS && This.buffers.length; //May not have started
			This.CLog("stop and start time difference" + (envInMS ? envInMS + "ms compensation" + This.envInFix + "ms" + " envIn:" + envInLen + " fps:" + (envInLen / envInMS * 1000).toFixed(1) : "-"));

			var end = function () {
				This._stop();//Completely turn off engineCtx
				if (autoClose) {
					This.close();
				};
			};
			var err = function (msg) {
				This.CLog("Failed to stop recording:" + msg, 1);
				False && False(msg);
				end();
			};
			var ok = function (blob, duration) {
				This.CLog("Stop recording encoding took" + (Date.now() - t1) + "ms audio duration" + duration + "ms file size" + blob.size + "b");
				if (set.takeoffEncodeChunk) {//If output is taken over, the blob length is 0 at this time
					This.CLog("When takeoffEncodeChunk is enabled, the blob returned by stop has a length of 0 and does not provide audio data", 3);
				} else if (blob.size < Math.max(100, duration / 2)) {//Is 1 second less than 0.5k?
					err("The generated" + set.type + "is invalid");
					return;
				};
				True && True(blob, duration);
				end();
			};
			if (!This.isMock) {
				var isCtxWait = This.state == 3;
				if (!This.state || isCtxWait) {
					err("Not started recording" + (isCtxWait ? "，AudioContext not running due to no user interaction before starting recording" : ""));
					return;
				};
				This._stop(true);
			};
			var size = This.recSize;
			if (!size) {
				err("No recording was collected");
				return;
			};
			if (!This.buffers[0]) {
				err("Audio buffers released");
				return;
			};
			if (!This[set.type]) {
				err("Encoder" + set.type + "not loaded");
				return;
			};

			//Environment configuration check, this is only for mock calls, because it has been checked by open
			if (This.isMock) {
				var checkMsg = This.envCheck(This.mockEnvInfo || { envName: "mock", canProcess: false });//Mock without environment information has no onProcess callback
				if (checkMsg) {
					err("Recording error:" + checkMsg);
					return;
				};
			};

			//This type has side-by-side transcoding (Worker) support
			var engineCtx = This.engineCtx;
			if (This[set.type + "_complete"] && engineCtx) {
				var duration = Math.round(engineCtx.pcmSize / set[sampleRateTxt] * 1000);//The length of the data after adoption and the length of the buffers may be slightly inconsistent, which is a precision problem of continuous sample rate conversion
				t1 = Date.now();
				This[set.type + "_complete"](engineCtx, function (blob) {
					ok(blob, duration);
				}, err);
				return;
			};

			//Standard UI thread transcoding, adjust sample rate
			t1 = Date.now();
			var chunk = Recorder.SampleData(This.buffers, This[srcSampleRateTxt], set[sampleRateTxt]);

			set[sampleRateTxt] = chunk[sampleRateTxt];
			var res = chunk.data;
			var duration = Math.round(res.length / set[sampleRateTxt] * 1000);

			This.CLog("Sampling" + size + "->" + res.length + " took:" + (Date.now() - t1) + "ms");

			setTimeout(function () {
				t1 = Date.now();
				This[set.type](res, function (blob) {
					ok(blob, duration);
				}, function (msg) {
					err(msg);
				});
			});
		}

	};

	if (window[RecTxt]) {
		CLog("Repeated introduction" + RecTxt, 3);
		window[RecTxt].Destroy();
	};
	window[RecTxt] = Recorder;


	//=======Extract pcm data from WebM byte stream, return Float32Array on success, null||-1 on failure=====
	var WebM_Extract = function (inBytes, scope) {
		if (!scope.pos) {
			scope.pos = [0]; scope.tracks = {}; scope.bytes = [];
		};
		var tracks = scope.tracks, position = [scope.pos[0]];
		var endPos = function () { scope.pos[0] = position[0] };

		var sBL = scope.bytes.length;
		var bytes = new Uint8Array(sBL + inBytes.length);
		bytes.set(scope.bytes); bytes.set(inBytes, sBL);
		scope.bytes = bytes;

		//First read the file header and Track information
		if (!scope._ht) {
			readMatroskaVInt(bytes, position);//EBML Header
			readMatroskaBlock(bytes, position);//Skip EBML Header content
			if (!BytesEq(readMatroskaVInt(bytes, position), [0x18, 0x53, 0x80, 0x67])) {
				return;//Segment not recognized
			}
			readMatroskaVInt(bytes, position);//Skip Segment length value
			while (position[0] < bytes.length) {
				var eid0 = readMatroskaVInt(bytes, position);
				var bytes0 = readMatroskaBlock(bytes, position);
				var pos0 = [0], audioIdx = 0;
				if (!bytes0) return;//Data is incomplete, waiting for buffering
				//Complete Track data, loop to read TrackEntry
				if (BytesEq(eid0, [0x16, 0x54, 0xAE, 0x6B])) {
					while (pos0[0] < bytes0.length) {
						var eid1 = readMatroskaVInt(bytes0, pos0);
						var bytes1 = readMatroskaBlock(bytes0, pos0);
						var pos1 = [0], track = { channels: 0, sampleRate: 0 };
						if (BytesEq(eid1, [0xAE])) {//TrackEntry
							while (pos1[0] < bytes1.length) {
								var eid2 = readMatroskaVInt(bytes1, pos1);
								var bytes2 = readMatroskaBlock(bytes1, pos1);
								var pos2 = [0];
								if (BytesEq(eid2, [0xD7])) {//Track Number
									var val = BytesInt(bytes2);
									track.number = val;
									tracks[val] = track;
								} else if (BytesEq(eid2, [0x83])) {//Track Type
									var val = BytesInt(bytes2);
									if (val == 1) track.type = "video";
									else if (val == 2) {
										track.type = "audio";
										if (!audioIdx) scope.track0 = track;
										track.idx = audioIdx++;
									} else track.type = "Type-" + val;
								} else if (BytesEq(eid2, [0x86])) {//Track Codec
									var str = "";
									for (var i = 0; i < bytes2.length; i++) {
										str += String.fromCharCode(bytes2[i]);
									}
									track.codec = str;
								} else if (BytesEq(eid2, [0xE1])) {
									while (pos2[0] < bytes2.length) {//Loop to read Audio properties
										var eid3 = readMatroskaVInt(bytes2, pos2);
										var bytes3 = readMatroskaBlock(bytes2, pos2);
										//Sample rate, bit depth, number of channels
										if (BytesEq(eid3, [0xB5])) {
											var val = 0, arr = new Uint8Array(bytes3.reverse()).buffer;
											if (bytes3.length == 4) val = new Float32Array(arr)[0];
											else if (bytes3.length == 8) val = new Float64Array(arr)[0];
											else CLog("WebM Track !Float", 1, bytes3);
											track[sampleRateTxt] = Math.round(val);
										} else if (BytesEq(eid3, [0x62, 0x64])) track.bitDepth = BytesInt(bytes3);
										else if (BytesEq(eid3, [0x9F])) track.channels = BytesInt(bytes3);
									}
								}
							}
						}
					};
					scope._ht = 1;
					CLog("WebM Tracks", tracks);
					endPos();
					break;
				}
			}
		}

		//Verify audio parameter information. If it does not meet the code requirements, all will be rejected.
		var track0 = scope.track0;
		if (!track0) return;
		if (track0.bitDepth == 16 && /FLOAT/i.test(track0.codec)) {
			track0.bitDepth = 32; //chrome v66 is actually a floating-point number
			CLog("WebM 16 changed to 32 bits", 3);
		}
		if (track0[sampleRateTxt] != scope[sampleRateTxt] || track0.bitDepth != 32 || track0.channels < 1 || !/(\b|_)PCM\b/i.test(track0.codec)) {
			scope.bytes = [];//The format is unexpected and cannot be processed, clear the buffered data
			if (!scope.bad) CLog("WebM Track unexpected", 3, scope);
			scope.bad = 1;
			return -1;
		}

		//Loop to read SimpleBlock in Cluster
		var datas = [], dataLen = 0;
		while (position[0] < bytes.length) {
			var eid1 = readMatroskaVInt(bytes, position);
			var bytes1 = readMatroskaBlock(bytes, position);
			if (!bytes1) break;//Data is incomplete, waiting for buffering
			if (BytesEq(eid1, [0xA3])) {//Complete SimpleBlock data
				var trackNo = bytes1[0] & 0xf;
				var track = tracks[trackNo];
				if (!track) {//Impossible to not have, data error?
					CLog("WebM !Track" + trackNo, 1, tracks);
				} else if (track.idx === 0) {
					var u8arr = new Uint8Array(bytes1.length - 4);
					for (var i = 4; i < bytes1.length; i++) {
						u8arr[i - 4] = bytes1[i];
					}
					datas.push(u8arr); dataLen += u8arr.length;
				}
			}
			endPos();
		}

		if (dataLen) {
			var more = new Uint8Array(bytes.length - scope.pos[0]);
			more.set(bytes.subarray(scope.pos[0]));
			scope.bytes = more; //Clear the buffered data that has been read
			scope.pos[0] = 0;

			var u8arr = new Uint8Array(dataLen); //Obtained audio data
			for (var i = 0, i2 = 0; i < datas.length; i++) {
				u8arr.set(datas[i], i2);
				i2 += datas[i].length;
			}
			var arr = new Float32Array(u8arr.buffer);

			if (track0.channels > 1) {//Multi-channel, extract one channel
				var arr2 = [];
				for (var i = 0; i < arr.length;) {
					arr2.push(arr[i]);
					i += track0.channels;
				}
				arr = new Float32Array(arr2);
			};
			return arr;
		}
	};
	//Are the contents of two byte arrays the same
	var BytesEq = function (bytes1, bytes2) {
		if (!bytes1 || bytes1.length != bytes2.length) return false;
		if (bytes1.length == 1) return bytes1[0] == bytes2[0];
		for (var i = 0; i < bytes1.length; i++) {
			if (bytes1[i] != bytes2[i]) return false;
		}
		return true;
	};
	//Convert byte array BE to int number
	var BytesInt = function (bytes) {
		var s = "";//0-8 bytes, js bitwise operation only supports 4 bytes
		for (var i = 0; i < bytes.length; i++) { var n = bytes[i]; s += (n < 16 ? "0" : "") + n.toString(16) };
		return parseInt(s, 16) || 0;
	};
	//Read a variable-length numeric byte array
	var readMatroskaVInt = function (arr, pos, trim) {
		var i = pos[0];
		if (i >= arr.length) return;
		var b0 = arr[i], b2 = ("0000000" + b0.toString(2)).substr(-8);
		var m = /^(0*1)(\d*)$/.exec(b2);
		if (!m) return;
		var len = m[1].length, val = [];
		if (i + len > arr.length) return;
		for (var i2 = 0; i2 < len; i2++) { val[i2] = arr[i]; i++; }
		if (trim) val[0] = parseInt(m[2] || '0', 2);
		pos[0] = i;
		return val;
	};
	//Read a content byte array with its own length
	var readMatroskaBlock = function (arr, pos) {
		var lenVal = readMatroskaVInt(arr, pos, 1);
		if (!lenVal) return;
		var len = BytesInt(lenVal);
		var i = pos[0], val = [];
		if (len < 0x7FFFFFFF) { //A very large value means no length
			if (i + len > arr.length) return;
			for (var i2 = 0; i2 < len; i2++) { val[i2] = arr[i]; i++; }
		}
		pos[0] = i;
		return val;
	};
	//=====End WebM read=====


	//Traffic statistics use 1-pixel image address, if set to empty, it will not participate in statistics
	Recorder.TrafficImgUrl = "//ia.51.la/go1?id=20469973&pvFlag=1";
	var Traffic = Recorder.Traffic = function (report) {
		report = report ? "/" + RecTxt + "/Report/" + report : "";
		var imgUrl = Recorder.TrafficImgUrl;
		if (imgUrl) {
			var data = Recorder.Traffic;
			var m = /^(https?:..[^\/#]*\/?)[^#]*/i.exec(location.href) || [];
			var host = (m[1] || "http://file/");
			var idf = (m[0] || host) + report;

			if (imgUrl.indexOf("//") == 0) {
				//Add http prefix to the url, if under file protocol, it cannot be used without a prefix
				if (/^https:/i.test(idf)) {
					imgUrl = "https:" + imgUrl;
				} else {
					imgUrl = "http:" + imgUrl;
				};
			};
			if (report) {
				imgUrl = imgUrl + "&cu=" + encodeURIComponent(host + report);
			};

			if (!data[idf]) {
				data[idf] = 1;

				var img = new Image();
				img.src = imgUrl;
				CLog("Traffic Analysis Image: " + (report || RecTxt + ".TrafficImgUrl=" + Recorder.TrafficImgUrl));
			};
		};
	};

}));